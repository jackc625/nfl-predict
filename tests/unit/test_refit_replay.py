"""Replay and verification stay whole across re-fit generations (Plan 37-17, Pitfall 10, R11, P6).

WHAT IS PROVEN
--------------
A fixture generation 1 is installed the way the re-fit hook installs one (Plan 37-10's fixture:
its models saved under new ids, its bet-rule record written by the real producer, its install
recorded in ``ledger/refit/dates.jsonl``), and the REAL ``record_forward_slate`` decides a fixture
week with it, writing the live generation-1 rows and their frozen row-19 twins.

* Replay re-derives a generation-1 row with ITS generation's bet rule, read from the ledger's own
  copy (``ledger/recipes/<sha>.json``) through ``load_bet_rule_record``: every store outside the
  ledger is renamed away first, so a pass shows the ledger's copies alone suffice (LDGR-09).
* A generation threshold that suppresses a target replays as suppressed: replay used the record's
  thresholds, not row-19's.
* A row-19 ledger replays exactly as before.

A full re-fit cycle then runs over a ledger seeded like the production one before Phase 37
(NULL-stamp weeks 3-4, row-19 weeks 5-6, all live): a generation-0 decision with its twins, the
install of generation 1, a generation-1 decision with its twins and a settle pass that grades. No
entry written before any step changes its seq, immutable values or chain hash (must-NOT P6), and
the verification CLI -- against a temporary repository whose ``ledger-anchor`` commits that
ledger's head, and the committed verdict-scope declaration -- passes over every row (R11).

Everything lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/``, ``logs/`` and ``ledger/`` are never read or written (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.neutral_hfa_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from backtest.shadow_pin_2026 import SHADOW_RECIPE
from backtest.weekly_bet_list import build_weekly_decision_bundle, load_bet_rule_record
from forward_ledger import runner
from forward_ledger.artifacts_copy import RECIPE_COPIES_DIRNAME
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.declarations import VERDICT_SCOPE_MODULE
from forward_ledger.replay import REPLAY_PASS, replay_ledger
from forward_ledger.schema import (
    ARM_LIVE,
    ARM_SHADOW,
    VERDICT_SCOPE_PRE_VERDICT,
    VERDICT_SCOPE_VERDICT,
)
from forward_ledger.settle import SILVER_GAMES_FILENAME, live_strategies
from forward_ledger.shadow import build_shadow_bundle
from forward_ledger.stamps import IN_FORCE_RECIPE_ID
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
    LedgerEntry,
    read_entries,
    verify_chain,
)
from pipeline.daily_steps import DailySlate
from refit.records import BET_RULES_DIRNAME, append_date_event, refit_dir
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_refit_decision_path import (
    ADMIT_NO_WP,
    GENERATION_ONE_IDS,
    InstalledFixture,
    attempt_event,
    build_stores,
    closed_event,
    install_generation_one,
    intent_event,
    ledger_rows,
    record,
    run_log_path,  # noqa: F401 -- the run-log redirect fixture, used by name
    write_generation_bet_rule,
    write_manifest_ids,
)
from tests.unit.test_replay_ledger import (
    DECIDED_AT,
    GAMES,
    LOCK,
    MODEL_IDS,
    SEASON,
    build_fixture_season,
)
from tests.unit.test_shadow_arm import Stores
from tests.unit.test_verify_ledger_cli import (
    PENDING,
    anchored_repo,
    cli_args,
    run_cli,
    write_ledger,
)


def _arm(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["arm"] == arm]


def _move_away(path: Path) -> None:
    """Rename *path* aside, so nothing can read it under its own name any more."""
    assert path.exists(), f"{path.as_posix()} was expected to exist before it is moved"
    path.rename(path.with_name(f"{path.name}.moved_away"))


def _remove_every_store_outside_the_ledger(
    installed: InstalledFixture, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Rename every store the decision read away, and the generation's own bet-rule directory.

    Left in place: the ledger's row file, its snapshots, and its artifact and recipe copies --
    exactly what the private backup holds (D-02). The working directory moves too, so a relative
    production default (``artifacts/``, ``outputs/``) resolves to nothing.
    """
    stores = installed.stores
    for path in (
        stores.artifacts,
        stores.gold,
        stores.silver,
        stores.manifests,
        stores.chain_fit,
        refit_dir(installed.ledger) / BET_RULES_DIRNAME,
    ):
        _move_away(path)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)


def _replay_statuses(ledger: Path) -> dict[tuple[Any, ...], str]:
    return {result.key: result.status for result in replay_ledger(ledger)}


# ---------------------------------------------------------------------------
# Task 1 (tracer): a generation-1 row replays with its own bet rule from the ledger alone
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("run_log_path")
def test_generation_one_rows_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = install_generation_one(tmp_path)
    record(installed.stores, installed.ledger)
    rows = ledger_rows(installed.ledger)
    live, shadow = _arm(rows, ARM_LIVE), _arm(rows, ARM_SHADOW)
    assert live, "the fixture week decided nothing, so nothing was replayed"
    assert {row["model_artifact_id"] for row in live} <= {
        GENERATION_ONE_IDS.model_id_for(target) for target in ("wp", "ats", "ou")
    }
    assert {row["recipe_id"] for row in shadow} == {SHADOW_RECIPE}
    assert len(shadow) == len(live)

    # The generation's bet rule is kept in the ledger like any chain-fit record (LDGR-09).
    sha = installed.bet_rule_sha256
    assert sha is not None
    copy = installed.ledger / RECIPE_COPIES_DIRNAME / f"{sha}.json"
    assert copy.is_file()

    _remove_every_store_outside_the_ledger(installed, monkeypatch, tmp_path)
    statuses = _replay_statuses(installed.ledger)

    assert len(statuses) == len(rows)
    failing = {key: status for key, status in statuses.items() if status != REPLAY_PASS}
    assert failing == {}


@pytest.mark.usefixtures("run_log_path")
def test_a_generation_record_threshold_is_used_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thresholds = {**EDGE_TIER_THRESHOLDS_BY_TARGET, "wp": ADMIT_NO_WP}
    installed = install_generation_one(
        tmp_path,
        edge_tier_thresholds={k: v for k, v in thresholds.items() if v is not None},
    )
    record(installed.stores, installed.ledger)
    rows = ledger_rows(installed.ledger)
    wp_live = [row for row in _arm(rows, ARM_LIVE) if row["target"] == "wp"]
    assert wp_live
    assert {row["status"] for row in wp_live} == {"suppressed"}
    row19_bets = [
        row
        for row in _arm(rows, ARM_SHADOW)
        if row["target"] == "wp" and row["status"] == "live"
    ]
    assert row19_bets, (
        "the row-19 rule bets no WP game, so the threshold was not tested"
    )

    _remove_every_store_outside_the_ledger(installed, monkeypatch, tmp_path)
    statuses = _replay_statuses(installed.ledger)

    # Every stored row replays -- the suppressed generation-1 WP rows included, which the row-19
    # thresholds would have re-derived as bets.
    failing = {key: status for key, status in statuses.items() if status != REPLAY_PASS}
    assert failing == {}
    assert len(statuses) == len(rows)


def test_row19_rows_replay_unchanged(tmp_path: Path) -> None:
    season = build_fixture_season(tmp_path)
    rows = ledger_rows(season.ledger)
    assert {row["model_artifact_id"] for row in rows} <= set(MODEL_IDS.values())

    # The row-19 copy still goes down the row-19 path: its overlay and the committed thresholds.
    copies = sorted((season.ledger / RECIPE_COPIES_DIRNAME).glob("*.json"))
    assert len(copies) == 1
    _fits, thresholds = load_bet_rule_record(copies[0])
    assert thresholds == dict(EDGE_TIER_THRESHOLDS_BY_TARGET)

    results = replay_ledger(season.ledger)

    assert [result.key[:5] for result in results] == [
        (row["game_id"], row["season"], row["week"], row["target"], row["arm"])
        for row in rows
    ]
    assert all(result.status == REPLAY_PASS for result in results)


# ---------------------------------------------------------------------------
# Task 2: a full re-fit cycle rewrites nothing and the verification CLI passes (P6, R11)
# ---------------------------------------------------------------------------

# Rows shaped like the production ledger before Phase 37: migrated weeks 3-4 rows carry NULL
# stamps and no snapshot; weeks 5-6 rows carry every row-19 stamp. Their games are not in the
# fixture schedule, so no decision of the cycle can collide with them.
_NULL_STAMP_GAMES: dict[int, tuple[str, ...]] = {
    3: ("2026_03_NE_NYG",),
    4: ("2026_04_TEN_HOU",),
}
_ROW19_GAMES: dict[int, tuple[str, ...]] = {
    5: ("2026_05_DEN_LV", "2026_05_CIN_PIT"),
    6: ("2026_06_MIN_DET", "2026_06_TB_NO"),
}

# (home, away) scores the settle pass grades from; every game of the cycle is played.
_SCORES: tuple[tuple[float, float], ...] = ((24.0, 17.0), (10.0, 20.0), (27.0, 23.0))
_SETTLED_AT = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class RefitCycle:
    """A ledger after a full re-fit cycle, and its entries as they stood before each step."""

    stores: Stores
    ledger: Path
    before_each_step: tuple[tuple[LedgerEntry, ...], ...]


def _pre_phase_items() -> list[tuple[str, dict[str, Any], Any]]:
    """The seeded rows: NULL-stamp weeks 3-4, then row-19 weeks 5-6, all live and pending."""
    null_stamp = [
        make_row(
            game_id=game_id,
            week=week,
            bet_side="home_cover",
            verdict_scope=VERDICT_SCOPE_PRE_VERDICT,
            **dict.fromkeys(LEDGER_STAMP_COLUMNS),
        )
        for week, games in _NULL_STAMP_GAMES.items()
        for game_id in games
    ]
    row19 = [
        make_row(
            game_id=game_id,
            week=week,
            bet_side="home_cover",
            recipe_id=IN_FORCE_RECIPE_ID,
            verdict_scope=VERDICT_SCOPE_VERDICT,
        )
        for week, games in _ROW19_GAMES.items()
        for game_id in games
    ]
    return [(ENTRY_KIND_ROW, row, PENDING) for row in (*null_stamp, *row19)]


def _decide(stores: Stores, ledger: Path, week: int) -> runner.RecordOutcome:
    """One real ``record_forward_slate`` of fixture *week*, the frozen arm on."""
    slate = DailySlate(
        run_date_et=LOCK[week].date(),
        lock=LOCK[week],
        schedule=pd.DataFrame(
            {
                "game_id": list(GAMES[week]),
                "season": [SEASON] * len(GAMES[week]),
                "week": [week] * len(GAMES[week]),
                "gameday": [date.isoformat(LOCK[week].date() + timedelta(days=1))]
                * len(GAMES[week]),
            }
        ),
    )
    return runner.record_forward_slate(
        slate,
        decided_at=DECIDED_AT[week],
        excluded_game_ids=frozenset(),
        publish_by=None,
        ledger_dir=ledger,
        artifacts_dir=stores.artifacts,
        gold_dir=stores.gold,
        silver_dir=stores.silver,
        manifest_dir=stores.manifests,
        bundle_builder=partial(
            build_weekly_decision_bundle, chain_fit_path=stores.chain_fit
        ),
        shadow_builder=partial(build_shadow_bundle, pin=stores.pin()),
    )


def _install_generation_one(stores: Stores, ledger: Path) -> None:
    """The fixture install of generation 1: its record, its closed install date, latest.json."""
    _path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)
    assert append_date_event(ledger, intent_event(GENERATION_ONE_IDS, sha))
    assert append_date_event(ledger, attempt_event("installed"))
    assert append_date_event(ledger, closed_event("ran"))
    write_manifest_ids(stores.artifacts, GENERATION_ONE_IDS)


def _record_scores(stores: Stores) -> None:
    """The schedule refresh: every game of the cycle, seeded ones included, now has a score."""
    path = stores.silver / SILVER_GAMES_FILENAME
    games = pd.read_parquet(path)
    seeded = pd.DataFrame(
        {
            "game_id": [
                game_id
                for weeks in (_NULL_STAMP_GAMES, _ROW19_GAMES)
                for games_of_week in weeks.values()
                for game_id in games_of_week
            ],
            "season": SEASON,
            "week": [
                week
                for weeks in (_NULL_STAMP_GAMES, _ROW19_GAMES)
                for week, games_of_week in weeks.items()
                for _game_id in games_of_week
            ],
        }
    )
    seeded["kickoff_et"] = games["kickoff_et"].iloc[0]
    games = pd.concat([games, seeded], ignore_index=True)
    scores = [_SCORES[index % len(_SCORES)] for index in range(len(games))]
    games["home_score"] = [home for home, _away in scores]
    games["away_score"] = [away for _home, away in scores]
    games.to_parquet(path, index=False)


def run_refit_cycle(root: Path) -> RefitCycle:
    """Seed the pre-phase rows, then decide, install, decide and settle over one ledger."""
    stores = build_stores(root)
    ledger = root / "ledger"
    write_ledger(ledger, _pre_phase_items())
    snapshots: list[tuple[LedgerEntry, ...]] = []

    snapshots.append(tuple(read_entries(ledger)))
    generation_zero = _decide(stores, ledger, 4)
    assert generation_zero.appended and generation_zero.shadow_appended

    snapshots.append(tuple(read_entries(ledger)))
    _install_generation_one(stores, ledger)
    generation_one = _decide(stores, ledger, 3)
    assert generation_one.appended and generation_one.shadow_appended

    snapshots.append(tuple(read_entries(ledger)))
    _record_scores(stores)
    settled = runner.settle_ledger(
        now=_SETTLED_AT,
        ledger_dir=ledger,
        silver_dir=stores.silver,
        manifest_dir=stores.manifests,
        strategies=live_strategies(stores.chain_fit),
    )
    assert settled.changed and settled.graded

    return RefitCycle(stores=stores, ledger=ledger, before_each_step=tuple(snapshots))


@pytest.mark.usefixtures("run_log_path")
def test_a_refit_cycle_rewrites_no_row(tmp_path: Path) -> None:
    cycle = run_refit_cycle(tmp_path)
    after = read_entries(cycle.ledger)

    assert verify_chain(after).ok
    for before in cycle.before_each_step:
        assert before, "a step started from an empty ledger, so nothing was protected"
        for old in before:
            new = after[old.seq]
            assert (new.seq, new.kind, new.immutable, new.chain_hash) == (
                old.seq,
                old.kind,
                old.immutable,
                old.chain_hash,
            ), f"seq {old.seq} was rewritten"

    rows = [entry.immutable for entry in after if entry.kind == ENTRY_KIND_ROW]
    recipes = {(row["arm"], row["recipe_id"]) for row in rows}
    assert {
        (ARM_LIVE, None),
        (ARM_LIVE, IN_FORCE_RECIPE_ID),
        (ARM_LIVE, REFIT_RECIPE_ID),
        (ARM_SHADOW, SHADOW_RECIPE),
    } <= recipes
    # The settle pass graded the seeded rows and the decided ones: only mutable halves moved.
    graded_weeks = {
        entry.immutable["week"]
        for entry in after
        if (entry.grading or {}).get("grading_status") not in (None, "pending")
    }
    assert {3, 4, 5, 6} <= graded_weeks


@pytest.mark.usefixtures("run_log_path", "isolated_git_env")
def test_the_verify_cli_passes_over_every_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cycle = run_refit_cycle(tmp_path)
    entries = read_entries(cycle.ledger)
    repo = anchored_repo(tmp_path / "public", entries)
    capsys.readouterr()

    code, out = run_cli(
        capsys, cli_args(cycle.ledger, repo), declaration=VERDICT_SCOPE_MODULE
    )

    lines = out.splitlines()
    assert "LOCAL_RESULT= PASS" in lines, out
    assert not [line for line in lines if line.startswith("FAILURE=")], out
    assert f"LEDGER_ENTRIES= {len(entries)}" in lines
    assert "VERDICT_DECLARED= True" in lines
    assert "VERDICT_STAMPS_OK= True" in lines
    assert "VERDICT_LABELS_OK= True" in lines
    regraded = next(line for line in lines if line.startswith("REGRADE_CHECKED= "))
    assert int(regraded.split()[1]) > 0
    assert code == 0, out
