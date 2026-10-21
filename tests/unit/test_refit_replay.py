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

Everything lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/``, ``logs/`` and ``ledger/`` are never read or written (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backtest.neutral_hfa_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from backtest.shadow_pin_2026 import SHADOW_RECIPE
from backtest.weekly_bet_list import load_bet_rule_record
from forward_ledger.artifacts_copy import RECIPE_COPIES_DIRNAME
from forward_ledger.replay import REPLAY_PASS, replay_ledger
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from refit.records import BET_RULES_DIRNAME, refit_dir
from tests.unit.test_refit_decision_path import (
    ADMIT_NO_WP,
    GENERATION_ONE_IDS,
    InstalledFixture,
    install_generation_one,
    ledger_rows,
    record,
    run_log_path,  # noqa: F401 -- the run-log redirect fixture, used by name
)
from tests.unit.test_replay_ledger import MODEL_IDS, build_fixture_season


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
