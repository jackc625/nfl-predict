"""The frozen shadow arm: the live decision re-decided with the pinned row-19 rule (Plan 37-01).

WHAT IS PROVEN
--------------
A fixture week is decided through the REAL decision path (``build_weekly_decision_bundle`` over
tiny real models saved under the row-19 ids, a synthetic gold, silver schedule and odds store, and
a temporary chain-fit record) and recorded by the real ``record_forward_slate`` with the shadow arm
on. Every new live row gets exactly one shadow twin stamped ``arm = shadow`` with the pinned ids and
``recipe-2026-row19-v1``; both arms land in ONE ledger replace, live rows first; each twin equals
its live row in every immutable column but ``arm`` (the A/A identity); and the live rows are
byte-identical to a run with the shadow arm off (D-04: nothing here changes what the live arm
decides).

Every store lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/``, ``logs/`` and ``ledger/`` are never read or written (COLD-05). The shadow pin used
here names the fixture chain-fit record; the committed pin is never touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.shadow_pin_2026 import SHADOW_RECIPE
from backtest.weekly_bet_list import build_weekly_decision_bundle
from data.upstream_live import AS_OF_ENV, LIVE_MANIFEST_SCHEMA_VERSION
from forward_ledger import run_log, runner
from forward_ledger.canonical import ENTRY_KIND_ROW, IMMUTABLE_COLUMNS_V1
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.shadow import ShadowPin, build_shadow_bundle
from forward_ledger.store import ledger_path, read_entries, verify_chain
from pipeline.daily_steps import DailySlate
from tests.fixtures.decision_frame import chain_fit_record
from tests.unit.test_forward_ledger_runner import _count_replaces
from tests.unit.test_replay_ledger import (
    BLEND_ID,
    CONVERTER_ID,
    DECIDED_AT,
    GAMES,
    LOCK,
    MODEL_IDS,
    SEASON,
    _write_artifacts,
    _write_gold,
    _write_silver,
)

WEEK = 4


# ---------------------------------------------------------------------------
# The fixture stores (tmp only), decided through the real path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stores:
    """The synthetic stores one decision reads, all under one temporary directory."""

    artifacts: Path
    gold: Path
    silver: Path
    manifests: Path
    chain_fit: Path

    def pin(self) -> ShadowPin:
        """The row-19 pin, naming this fixture's chain-fit record and its sha256."""
        return ShadowPin(
            model_ids=MODEL_IDS,
            blend=BLEND_ID,
            converter=CONVERTER_ID,
            recipe=SHADOW_RECIPE,
            chain_fit_path=str(self.chain_fit),
            chain_fit_sha256=hashlib.sha256(self.chain_fit.read_bytes()).hexdigest(),
        )


def _write_manifest(directory: Path) -> None:
    directory.mkdir()
    manifest = {
        "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
        "season": SEASON,
        "datasets": {
            "schedules": {
                "captures": [
                    {
                        "week": WEEK,
                        "sequence": 1,
                        "sha256": "s" * 64,
                        "path": "bronze/s",
                        "captured_at_utc": "2026-10-03T19:00:00+00:00",
                    }
                ]
            }
        },
    }
    (directory / f"{SEASON}.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> Stores:
    root = tmp_path_factory.mktemp("shadow_arm_stores")
    built = Stores(
        artifacts=root / "artifacts",
        gold=root / "gold",
        silver=root / "silver",
        manifests=root / "upstream_live",
        chain_fit=root / "chain_fit.json",
    )
    built.artifacts.mkdir()
    _write_artifacts(built.artifacts)
    _write_gold(built.gold)
    _write_silver(built.silver)
    _write_manifest(built.manifests)
    built.chain_fit.write_text(json.dumps(chain_fit_record(0.0)), encoding="utf-8")
    return built


@pytest.fixture
def run_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the ledger run log into *tmp_path*; production ``logs/`` is never written."""
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    path = tmp_path / "logs" / "ledger_runs.jsonl"
    monkeypatch.setattr(run_log, "LEDGER_RUN_LOG", path)
    return path


def _slate() -> DailySlate:
    return DailySlate(
        run_date_et=date(2026, 10, 3),
        lock=LOCK[WEEK],
        schedule=pd.DataFrame(
            {
                "game_id": list(GAMES[WEEK]),
                "season": [SEASON] * len(GAMES[WEEK]),
                "week": [WEEK] * len(GAMES[WEEK]),
                "gameday": ["2026-10-04"] * len(GAMES[WEEK]),
            }
        ),
    )


def _record(stores: Stores, ledger: Path, **kwargs: Any) -> runner.RecordOutcome:
    """One real ``record_forward_slate`` call over the fixture stores, shadow arm on."""
    options: dict[str, Any] = {
        "decided_at": DECIDED_AT[WEEK],
        "excluded_game_ids": frozenset(),
        "publish_by": None,
        "ledger_dir": ledger,
        "artifacts_dir": stores.artifacts,
        "gold_dir": stores.gold,
        "silver_dir": stores.silver,
        "manifest_dir": stores.manifests,
        "bundle_builder": partial(
            build_weekly_decision_bundle, chain_fit_path=stores.chain_fit
        ),
        "shadow_builder": partial(build_shadow_bundle, pin=stores.pin()),
    }
    options.update(kwargs)
    return runner.record_forward_slate(_slate(), **options)


def _rows(ledger: Path) -> list[dict[str, Any]]:
    return [
        entry.immutable
        for entry in read_entries(ledger)
        if entry.kind == ENTRY_KIND_ROW
    ]


def _twin_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["game_id"], row["season"], row["week"], row["target"])


def _arm(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["arm"] == arm]


# ---------------------------------------------------------------------------
# Task 1: one path end to end -- pin, shared re-score, shadow decision, one commit
# ---------------------------------------------------------------------------


def test_shadow_twins_written_in_one_commit(
    stores: Stores,
    tmp_path: Path,
    run_log_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = tmp_path / "ledger"
    replaced = _count_replaces(monkeypatch)

    outcome = _record(stores, ledger)

    entries = read_entries(ledger)
    assert verify_chain(entries).ok
    rows = _rows(ledger)
    live, shadow = _arm(rows, ARM_LIVE), _arm(rows, ARM_SHADOW)
    assert live, "the fixture week decided nothing, so nothing was tested"
    assert {row["status"] for row in live} == {"live", "suppressed"}
    # Live rows first, then shadow rows: the documented fixed order.
    assert [row["arm"] for row in rows] == [ARM_LIVE] * len(live) + [ARM_SHADOW] * len(
        shadow
    )
    # Exactly one shadow twin per live (game_id, season, week, target).
    assert sorted(_twin_key(row) for row in shadow) == sorted(
        _twin_key(row) for row in live
    )
    for row in shadow:
        assert row["model_artifact_id"] == MODEL_IDS[row["target"]]
        assert row["blend_id"] == BLEND_ID
        assert row["recipe_id"] == SHADOW_RECIPE
    assert replaced == [ledger_path(ledger)], "both arms did not land in ONE replace"
    assert (outcome.appended, outcome.shadow_appended) == (len(live), len(shadow))
    assert outcome.shadow_holes == ()


def test_aa_twins_are_identical_outside_the_exclusion_list(
    stores: Stores, tmp_path: Path, run_log_path: Path
) -> None:
    ledger = tmp_path / "ledger"
    _record(stores, ledger)

    rows = _rows(ledger)
    live = {_twin_key(row): row for row in _arm(rows, ARM_LIVE)}
    shadow = _arm(rows, ARM_SHADOW)
    assert shadow
    compared = [name for name in IMMUTABLE_COLUMNS_V1 if name != "arm"]
    for twin in shadow:
        original = live[_twin_key(twin)]
        differing = [name for name in compared if twin[name] != original[name]]
        assert differing == [], (_twin_key(twin), differing)


def test_live_rows_byte_identical_with_shadow_on_and_off(
    stores: Stores, tmp_path: Path, run_log_path: Path
) -> None:
    with_shadow = tmp_path / "with_shadow"
    without_shadow = tmp_path / "without_shadow"

    on = _record(stores, with_shadow)
    off = _record(stores, without_shadow, shadow_builder=None)

    lines_on = ledger_path(with_shadow).read_bytes().splitlines()
    lines_off = ledger_path(without_shadow).read_bytes().splitlines()
    assert on.appended == off.appended == len(lines_off)
    assert off.shadow_appended == 0
    assert len(lines_on) == 2 * len(lines_off)
    assert lines_on[: len(lines_off)] == lines_off
