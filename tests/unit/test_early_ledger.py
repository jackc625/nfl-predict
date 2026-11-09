"""Writing the early-week arm's rows to the forward ledger (backtest/early_arm_2026.py).

WHAT IS PROVEN
--------------
``forward_ledger.runner.record_early_week`` writes one week's early decisions through the same
append-only path the daily run uses, with no shadow arm:

* every row is stamped ``arm = early``, ``recipe-2026-v2`` and ``fill-v2``, and is ``pre_verdict``
  even in week 11, which IS inside the committed verdict weeks -- so it can never count;
* a ledger that already holds verdict rows, which production does, does not refuse the write;
* the same decision again writes nothing; a different pick for a stored key is refused by name;
* nothing lands after the deadline, and models that are not the v2 generation's are refused;
* the record's hash handed in as ``meta_extra`` reaches the stored snapshot and moves its digest;
* an early bet grades at the price it was struck at.

The bundle carries ``fits={}``, the shape the early decision hands over, so that shape is proven
through the snapshot and the record path. The installed-generation history is written by the real
record helpers. Every store lives under pytest's temporary directories: production ``ledger/``,
``artifacts/``, ``data/`` and ``logs/`` are never read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.weekly_bet_list import DecisionBundle, PublishDeadlinePassedError
from data.upstream_live import AS_OF_ENV, LIVE_MANIFEST_SCHEMA_VERSION
from forward_ledger.canonical import (
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    IMMUTABLE_COLUMNS_V1,
)
from forward_ledger.declarations import load_verdict_scope, verdict_scope_label
from forward_ledger.runner import RecordOutcome, record_early_week
from forward_ledger.schema import BET_LIST_GRADING_COLUMNS
from forward_ledger.settle import grading_updates
from forward_ledger.snapshots import (
    SNAPSHOTS_DIRNAME,
    compute_snapshot_digest,
    load_decision_snapshot,
)
from forward_ledger.store import (
    build_entry,
    commit_changes,
    ledger_path,
    read_entries,
    verify_chain,
)
from forward_ledger.verdict import verdict_rows
from models.artifacts import ResolvedArtifacts
from refit.generations import UnknownGenerationError, generation_zero_ids
from tests.unit.test_forward_ledger_store import key_of, make_row
from tests.unit.test_ledger_settle import STRATEGIES
from tests.unit.test_owner_install_records import V2_IDS, ledger_with_owner_install
from tests.unit.test_refit_decision_path import GENERATION_ONE_IDS

SEASON = 2026
# Week 11 is inside the committed verdict weeks, so a row counts unless the stamp call stops it.
WEEK = 11
# Tuesday 2026-11-17 17:00 ET, the arm's first decision.
DECIDED_AT = datetime(2026, 11, 17, 22, 0, tzinfo=UTC)
# Saturday 2026-11-21 18:00 ET, the lock of a Sunday game.
FREEZE_TS = "2026-11-21T18:00:00-05:00"

OU_GAME = "2026_11_KC_BUF"
ATS_GAME = "2026_11_DAL_PHI"
META_EXTRA: dict[str, str] = {"early_arm_record_sha256": "a" * 64}

# The first 23 immutable columns: what a decision emits. The stamps are the stamping site's alone.
_DECISION_COLUMNS: tuple[str, ...] = IMMUTABLE_COLUMNS_V1[:23]
_MODEL_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

_OTHER_IDS = ResolvedArtifacts(
    wp="wp_20990101_000001",
    ats="ats_20990101_000002",
    ou="ou_20990101_000003",
    blend="blend_20990101_000004",
    converter=V2_IDS.converter,
)


# ---------------------------------------------------------------------------
# Fixture stores and the decision bundle (tmp_path only)
# ---------------------------------------------------------------------------


@dataclass
class _Stores:
    """The stores one early write reads, and the run-log events it emitted."""

    ledger: Path
    artifacts: Path
    gold: Path
    manifests: Path
    chain_fit: Path
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def named(self, event: str) -> list[dict[str, Any]]:
        return [fields for name, fields in self.events if name == event]


def _gold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [OU_GAME, ATS_GAME],
            "season": [SEASON, SEASON],
            "week": [WEEK, WEEK],
            "elo_diff": [12.5, -3.0],
        }
    )


def _write_artifacts(artifacts: Path) -> None:
    """One tiny directory per id the v2 generation names: the copy step needs nothing more."""
    for artifact_id in (
        V2_IDS.wp,
        V2_IDS.ats,
        V2_IDS.ou,
        V2_IDS.blend,
        V2_IDS.converter,
    ):
        directory = artifacts / str(artifact_id)
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps({"id": artifact_id}), encoding="utf-8"
        )


def _write_manifest(manifests: Path) -> None:
    manifests.mkdir()
    capture = {
        "week": WEEK,
        "sequence": 1,
        "sha256": "s" * 64,
        "path": "bronze/s",
        "captured_at_utc": "2026-11-17T21:30:00+00:00",
    }
    (manifests / f"{SEASON}.json").write_text(
        json.dumps(
            {
                "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
                "season": SEASON,
                "datasets": {"schedules": {"captures": [capture]}},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Stores:
    """Generation 2 installed as the v2 recipe, plus tiny artifacts, gold and a live manifest."""
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    made = _Stores(
        ledger=ledger_with_owner_install(tmp_path, installed=True),
        artifacts=tmp_path / "artifacts",
        gold=tmp_path / "gold",
        manifests=tmp_path / "upstream_live",
        chain_fit=tmp_path / "early_rule.json",
    )
    _write_artifacts(made.artifacts)
    made.gold.mkdir()
    for target in _MODEL_TARGETS:
        _gold().to_parquet(made.gold / f"features_{target}.parquet", index=False)
    _write_manifest(made.manifests)
    made.chain_fit.write_bytes(b'{"record_id": "synthetic_early_rule"}\n')
    return made


def _early_row(
    game_id: str, target: str, decided_at: datetime, **overrides: Any
) -> dict[str, Any]:
    """One decided early row as the decision emits it: priced at a named book, no stamp yet."""
    fields: dict[str, Any] = {
        "game_id": game_id,
        "week": WEEK,
        "target": target,
        "bet_side": "over" if target == "ou" else "home_cover",
        "line": 44.5 if target == "ou" else -2.5,
        "slipped_line": 44.5 if target == "ou" else -2.5,
        "selected_odds": -105.0,
        "eligibility_label": "book:fanduel",
        "snapshot_ts": "2026-11-17T16:40:00-05:00",
        "freeze_ts": FREEZE_TS,
        "decided_at_utc": decided_at.isoformat(),
        **overrides,
    }
    row = make_row(**fields)
    return {name: row[name] for name in _DECISION_COLUMNS}


def _bundle(
    stores: _Stores,
    decided_at: datetime = DECIDED_AT,
    *,
    ids: ResolvedArtifacts = V2_IDS,
    ou_overrides: dict[str, Any] | None = None,
) -> DecisionBundle:
    """Two early bets, with ``fits={}``: the early decision has no chain fit to hand over."""
    games = [OU_GAME, ATS_GAME]
    return DecisionBundle(
        frame=pd.DataFrame(
            [
                _early_row(OU_GAME, "ou", decided_at, **(ou_overrides or {})),
                _early_row(ATS_GAME, "ats", decided_at),
            ]
        ),
        candidates=pd.DataFrame(
            {"game_id": games, "target": ["ou", "ats"], "model_value": [47.1, -3.4]}
        ),
        schedule=pd.DataFrame(
            {
                "game_id": games,
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "gameday": ["2026-11-22", "2026-11-22"],
            }
        ),
        gold_inputs={target: _gold() for target in _MODEL_TARGETS},
        fits={},
        resolved=ids,
        chain_fit_path=stores.chain_fit,
        run_instant=decided_at,
    )


def _odds() -> pd.DataFrame:
    """The recorded board the decision read: each game at one book, before the decision instant."""
    return pd.DataFrame(
        {
            "game_id": [OU_GAME, ATS_GAME],
            "sportsbook": ["fanduel", "fanduel"],
            "snapshot_ts": [pd.Timestamp("2026-11-17T21:40:00Z")] * 2,
            "created_at": [pd.Timestamp("2026-11-17T21:40:00Z")] * 2,
            "spread": [-3.0, -2.5],
            "total": [44.5, 41.0],
        }
    )


def _record(
    stores: _Stores, bundle: DecisionBundle | None = None, **overrides: Any
) -> RecordOutcome:
    options: dict[str, Any] = {
        "odds": _odds(),
        "gold_dir": stores.gold,
        "decided_at": DECIDED_AT,
        "publish_by": None,
        "meta_extra": META_EXTRA,
        "ledger_dir": stores.ledger,
        "artifacts_dir": stores.artifacts,
        "manifest_dir": stores.manifests,
        "log": stores.log,
    }
    options.update(overrides)
    return record_early_week(bundle or _bundle(stores), **options)


def _snapshot_dirs(ledger: Path) -> list[str]:
    root = ledger / SNAPSHOTS_DIRNAME
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


# ---------------------------------------------------------------------------
# The stamps, and why an early row can never count
# ---------------------------------------------------------------------------


def test_early_rows_are_stamped_early_and_never_count(stores: _Stores) -> None:
    """Week 11 is inside the declared verdict weeks, so only the stamp call keeps it out."""
    scope = load_verdict_scope()
    assert verdict_scope_label(SEASON, WEEK, scope) == "verdict", (
        "week 11 must be a counted week, or pre_verdict below proves nothing"
    )

    outcome = _record(stores)

    assert (outcome.decided, outcome.appended, outcome.wrote) == (2, 2, True)
    row = outcome.appended_rows[0]
    assert row["arm"] == "early" and row["verdict_scope"] == "pre_verdict"
    assert (
        row["recipe_id"] == "recipe-2026-v2" and row["fill_convention_id"] == "fill-v2"
    )
    assert row["week"] == WEEK
    assert row["eligibility_label"] == "book:fanduel"
    assert row["selected_odds"] == -105.0

    entries = read_entries(stores.ledger)
    assert verify_chain(entries).ok
    assert {entry.immutable["arm"] for entry in entries} == {"early"}
    assert {entry.immutable["verdict_scope"] for entry in entries} == {"pre_verdict"}
    assert {
        entry.immutable["target"]: entry.immutable["model_artifact_id"]
        for entry in entries
    } == {"ou": V2_IDS.ou, "ats": V2_IDS.ats}
    assert {entry.immutable["blend_id"] for entry in entries} == {V2_IDS.blend}
    assert verdict_rows(entries, scope).empty

    (digest,) = _snapshot_dirs(stores.ledger)
    assert row["decision_snapshot_digest"] == digest
    for artifact_id in (V2_IDS.wp, V2_IDS.blend, V2_IDS.converter):
        assert (stores.ledger / "artifacts" / str(artifact_id)).is_dir()
    assert (stores.ledger / "recipes").is_dir()
    (append,) = stores.named("append")
    assert append["appended_rows"] == 2 and append["snapshot_digest"] == digest


def test_a_ledger_holding_verdict_rows_does_not_refuse_an_early_row(
    stores: _Stores,
) -> None:
    """Production's ledger holds verdict rows: an early write states it makes none of its own."""
    commit_changes(stores.ledger, new_rows=[make_row(verdict_scope="verdict")])

    outcome = _record(stores)

    assert outcome.appended == 2
    assert {row["verdict_scope"] for row in outcome.appended_rows} == {"pre_verdict"}


# ---------------------------------------------------------------------------
# First pick stands
# ---------------------------------------------------------------------------


def test_the_same_decision_again_writes_nothing(stores: _Stores) -> None:
    _record(stores)
    before = ledger_path(stores.ledger).read_bytes()
    snapshots = _snapshot_dirs(stores.ledger)
    later = DECIDED_AT + timedelta(hours=1)

    outcome = _record(stores, _bundle(stores, later), decided_at=later)

    assert outcome.appended == 0
    assert outcome.identical == 2
    assert outcome.wrote is False
    assert outcome.appended_rows == ()
    assert ledger_path(stores.ledger).read_bytes() == before
    assert _snapshot_dirs(stores.ledger) == snapshots


def test_a_different_pick_for_a_stored_key_is_refused_by_name(
    stores: _Stores, capsys: pytest.CaptureFixture[str]
) -> None:
    _record(stores)
    before = ledger_path(stores.ledger).read_bytes()

    outcome = _record(stores, _bundle(stores, ou_overrides={"bet_side": "under"}))

    assert outcome.appended == 0 and outcome.wrote is False
    assert outcome.identical == 1
    ((key, fields),) = outcome.refused
    assert key == (OU_GAME, SEASON, WEEK, "ou", "early")
    assert fields == ("bet_side",)
    assert ledger_path(stores.ledger).read_bytes() == before
    assert "LEDGER_REFUSED=" in capsys.readouterr().out
    assert stores.named("first_pick_refusal") == [
        {"key": list(key), "fields": ["bet_side"]}
    ]


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_nothing_lands_after_the_deadline(stores: _Stores) -> None:
    deadline = DECIDED_AT + timedelta(minutes=5)

    with pytest.raises(PublishDeadlinePassedError):
        _record(
            stores,
            publish_by=deadline,
            clock=lambda: deadline + timedelta(seconds=1),
        )

    assert not (stores.ledger / SNAPSHOTS_DIRNAME).exists()
    assert not ledger_path(stores.ledger).exists()
    assert not (stores.ledger / "artifacts").exists()


@pytest.mark.parametrize(
    "ids",
    [GENERATION_ONE_IDS, generation_zero_ids(), _OTHER_IDS],
    ids=["the_refit_generation", "the_frozen_pin", "no_generation_at_all"],
)
def test_models_that_are_not_v2_are_refused(
    stores: _Stores, ids: ResolvedArtifacts
) -> None:
    with pytest.raises(UnknownGenerationError):
        _record(stores, _bundle(stores, ids=ids))

    assert not ledger_path(stores.ledger).exists()
    assert not (stores.ledger / SNAPSHOTS_DIRNAME).exists()


# ---------------------------------------------------------------------------
# The record's hash in the snapshot
# ---------------------------------------------------------------------------


def test_the_record_hash_reaches_the_snapshot_and_moves_its_digest(
    stores: _Stores,
) -> None:
    bundle = _bundle(stores)

    outcome = _record(stores, bundle)

    digest = outcome.appended_rows[0]["decision_snapshot_digest"]
    assert load_decision_snapshot(stores.ledger, digest).meta["extra"] == META_EXTRA
    assert digest == compute_snapshot_digest(bundle, meta_extra=META_EXTRA)
    assert digest != compute_snapshot_digest(
        bundle, meta_extra={"early_arm_record_sha256": "b" * 64}
    )
    assert digest != compute_snapshot_digest(bundle)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_an_early_bet_grades_at_its_own_price() -> None:
    """Over 44.5 at -105 with 50 scored pays 100/105, never the -110 the old rule assumed."""
    row = make_row(
        game_id=OU_GAME,
        week=WEEK,
        target="ou",
        bet_side="over",
        line=44.5,
        slipped_line=44.5,
        selected_odds=-105.0,
        eligibility_label="book:fanduel",
        arm="early",
        recipe_id="recipe-2026-v2",
        fill_convention_id="fill-v2",
        verdict_scope="pre_verdict",
    )
    entry = build_entry(
        GENESIS_HASH,
        0,
        ENTRY_KIND_ROW,
        row,
        grading={
            **dict.fromkeys(BET_LIST_GRADING_COLUMNS),
            "grading_status": "pending",
        },
    )
    realized = {"ou": {OU_GAME: 50.0}, "ats": {OU_GAME: 7.0}}

    updates = grading_updates(
        [entry], STRATEGIES, realized, datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    )

    values = updates[key_of(row)]
    assert values["grading_status"] == "win"
    assert values["payout_flat"] == pytest.approx(100 / 105)
