"""The go-live migration of the old writer's forward rows into the ledger (Plan 34-12 Task 1, D-14).

Every 2026 forward row in ``outputs/bet_list/bet_list.parquet`` moves into the ledger AS STORED, in
stored order, as the first entries chained from ``GENESIS_HASH``: ``arm='live'``, every stamp
NULL, ``pre_verdict``, ``bootstrap_regime`` on weeks 2-4 only, the grading half as stored and the
fill and closing halves NULL. The outputs pair is then rewritten with the 2025 replay rows only --
after the ledger write is verified, never before. A non-empty ledger and a payload carrying a stamp
are refused by name; ``--finish`` completes an interrupted rewrite only after proving the ledger
holds exactly the stored forward rows.

Every test builds a FIXTURE parquet under ``tmp_path`` in the production file's 29-column shape;
nothing here reads or writes ``outputs/``, ``ledger/`` or ``logs/`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache import (
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
)
from backtest.weekly_bet_list import (
    BET_LIST_ARTIFACT_NAME,
    BET_TRACKER_ARTIFACT_NAME,
    read_bet_list_artifact,
    write_bet_list_pair,
)
from forward_ledger.canonical import (
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    canonical_entry_bytes,
    canonical_values,
    chain_hash,
)
from forward_ledger.migration import (
    AlreadyMigratedError,
    MigrationMismatchError,
    MigrationStampError,
    build_migrated_entries,
    commit_migrated_rows,
    migrate_forward_rows,
)
from forward_ledger.schema import LEDGER_ROW_KEY
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
    ledger_path,
    read_entries,
    verify_chain,
)
from scripts import migrate_forward_rows as cli

# The production file's shape: the 23 pre-Phase-34 immutable columns, then the 6 grading columns.
OLD_IMMUTABLE: tuple[str, ...] = BET_LIST_IMMUTABLE_COLUMNS[
    : BET_LIST_IMMUTABLE_COLUMNS.index("decided_at_utc") + 1
]
OLD_COLUMNS: list[str] = [*OLD_IMMUTABLE, *BET_LIST_GRADING_COLUMNS]

GRADED_AT = pd.Timestamp("2026-10-05T03:14:15.926535+00:00")


def _row(
    *,
    provenance: str,
    season: int,
    week: int,
    game: str,
    target: str = "ats",
    status: str = "live",
    grading_status: str = "pending",
) -> dict[str, Any]:
    """One stored bet-list row in the 29-column shape the old writer wrote."""
    forward = provenance == PROVENANCE_FORWARD
    is_live = status == "live"
    settled = grading_status != "pending"
    return {
        "game_id": f"{season}_{week:02d}_{game}",
        "season": season,
        "week": week,
        "target": target,
        "bet_side": "home_cover" if is_live else None,
        "model_value": -3.25 if is_live else float("nan"),
        "market_value": -2.5,
        "line": -2.5,
        "slipped_line": -3.0 if is_live else float("nan"),
        "calibrated_p_side": 0.5431 if is_live else float("nan"),
        "per_bet_ev": 0.0377 if is_live else float("nan"),
        "stake_units": 1.25 if is_live else float("nan"),
        "ev_tier": "medium" if is_live else None,
        "status": status,
        "rejection_reason": None if is_live else "no_market_line",
        "eligibility_label": "no_subpopulation",
        "snapshot_ts": f"{season}-10-01T17:00:48.334399-04:00" if forward else None,
        "freeze_ts": f"{season}-10-01T18:00:00-04:00",
        "selected_odds": -110.0 if is_live else float("nan"),
        "flat_stake": 1.0 if is_live else float("nan"),
        "provenance": provenance,
        "validation_type": "forward_realized" if forward else "clean_holdout",
        "decided_at_utc": f"{season}-10-01T21:14:37.706834+00:00" if forward else None,
        "grading_status": grading_status,
        "outcome": {"win": True, "loss": False, "push": None, "pending": None}[
            grading_status
        ],
        "clv": 0.0123 if is_live else float("nan"),
        "payout_flat": {"win": 0.9090909090909091, "loss": -1.0}.get(
            grading_status, float("nan")
        ),
        "realized_units": {"win": 1.1363636363636365, "loss": -1.25}.get(
            grading_status, float("nan")
        ),
        "graded_at": GRADED_AT if settled else pd.NaT,
    }


def _stored_rows() -> list[dict[str, Any]]:
    """Replay 2025 rows INTERLEAVED with forward 2026 weeks 3, 4 and 5, in that file order."""
    replay, forward = PROVENANCE_BACKTEST_REPLAY, PROVENANCE_FORWARD
    return [
        _row(
            provenance=replay, season=2025, week=1, game="KC_BAL", grading_status="win"
        ),
        _row(
            provenance=forward, season=2026, week=3, game="NYJ_NE", status="suppressed"
        ),
        _row(
            provenance=forward,
            season=2026,
            week=3,
            game="NYJ_NE",
            target="wp",
            status="suppressed",
        ),
        _row(
            provenance=replay,
            season=2025,
            week=2,
            game="BUF_MIA",
            grading_status="loss",
        ),
        _row(
            provenance=forward, season=2026, week=4, game="ATL_NO", grading_status="win"
        ),
        _row(
            provenance=forward,
            season=2026,
            week=4,
            game="DAL_NYG",
            grading_status="loss",
        ),
        _row(provenance=forward, season=2026, week=4, game="SF_LA", target="ou"),
        _row(
            provenance=replay, season=2025, week=3, game="DET_GB", status="suppressed"
        ),
        _row(provenance=forward, season=2026, week=5, game="KC_JAX"),
    ]


def _write_fixture(output_dir: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    """The fixture bet list as a 29-column parquet, plus a stale tracker beside it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows or _stored_rows(), columns=OLD_COLUMNS)
    frame["graded_at"] = pd.to_datetime(frame["graded_at"], utc=True)
    path = output_dir / BET_LIST_ARTIFACT_NAME
    frame.to_parquet(path, index=False)
    (output_dir / BET_TRACKER_ARTIFACT_NAME).write_text("[]", encoding="utf-8")
    return path


def _stored_frame(output_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(output_dir / BET_LIST_ARTIFACT_NAME)


def _forward(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (frame["provenance"] == PROVENANCE_FORWARD) & (frame["season"] == 2026)
    return frame[mask].reset_index(drop=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Log:
    """Collects ``(event, fields)`` instead of writing ``logs/ledger_runs.jsonl``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event: str, **fields: Any) -> bool:
        self.events.append((event, fields))
        return True


@pytest.fixture
def stores(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "outputs" / "bet_list"
    _write_fixture(output_dir)
    return output_dir, tmp_path / "ledger"


def _apply(output_dir: Path, ledger_dir: Path) -> _Log:
    log = _Log()
    migrate_forward_rows(output_dir, ledger_dir, apply=True, log=log)
    return log


# ---------------------------------------------------------------------------
# The migrated entries
# ---------------------------------------------------------------------------


def test_migrates_forward_2026_rows_in_stored_order(stores: tuple[Path, Path]) -> None:
    output_dir, ledger_dir = stores
    stored_forward = _forward(_stored_frame(output_dir))
    _apply(output_dir, ledger_dir)

    entries = read_entries(ledger_dir)
    assert [entry.seq for entry in entries] == list(range(len(stored_forward)))
    assert all(entry.kind == ENTRY_KIND_ROW for entry in entries)
    stored_keys = [
        (row.game_id, row.season, row.week, row.target, "live")
        for row in stored_forward.itertuples()
    ]
    assert [tuple(e.immutable[name] for name in LEDGER_ROW_KEY) for e in entries] == (
        stored_keys
    )
    first = entries[0]
    assert first.chain_hash == chain_hash(
        GENESIS_HASH, canonical_entry_bytes(ENTRY_KIND_ROW, first.immutable)
    )
    assert verify_chain(entries).ok


def test_migrated_rows_have_null_stamps_and_pre_verdict(
    stores: tuple[Path, Path],
) -> None:
    output_dir, ledger_dir = stores
    _apply(output_dir, ledger_dir)

    entries = read_entries(ledger_dir)
    assert entries
    for entry in entries:
        immutable = entry.immutable
        assert all(immutable[name] is None for name in LEDGER_STAMP_COLUMNS)
        assert immutable["arm"] == "live"
        assert immutable["verdict_scope"] == "pre_verdict"
        expected_regime = "bootstrap_regime" if immutable["week"] in (3, 4) else None
        assert immutable["regime_label"] == expected_regime
        assert entry.fill is not None
        assert all(value is None for value in entry.fill.values())
        assert entry.closing is not None
        assert all(value is None for value in entry.closing.values())
    assert {
        e.immutable["week"] for e in entries if e.immutable["regime_label"] is None
    } == {5}


def test_migrated_values_are_as_stored(stores: tuple[Path, Path]) -> None:
    output_dir, ledger_dir = stores
    stored_forward = _forward(_stored_frame(output_dir))
    _apply(output_dir, ledger_dir)

    entries = read_entries(ledger_dir)
    for entry, stored in zip(entries, stored_forward.to_dict("records"), strict=True):
        full = {
            **{name: stored[name] for name in OLD_IMMUTABLE},
            **{name: entry.immutable[name] for name in BET_LIST_IMMUTABLE_COLUMNS[23:]},
        }
        expected = canonical_values(ENTRY_KIND_ROW, full)
        assert {name: entry.immutable[name] for name in OLD_IMMUTABLE} == {
            name: expected[name] for name in OLD_IMMUTABLE
        }
        assert entry.immutable["decided_at_utc"] == stored["decided_at_utc"]
        assert entry.immutable["validation_type"] == "forward_realized"

        grading = entry.grading
        assert grading is not None
        assert grading["grading_status"] == stored["grading_status"]
        assert grading["outcome"] == stored["outcome"]
        for name in ("clv", "payout_flat", "realized_units"):
            value = stored[name]
            assert grading[name] == (None if pd.isna(value) else value)
        if pd.isna(stored["graded_at"]):
            assert grading["graded_at"] is None
        else:
            assert grading["graded_at"] == stored["graded_at"].isoformat()
            assert pd.Timestamp(grading["graded_at"]) == stored["graded_at"]


# ---------------------------------------------------------------------------
# The outputs rewrite
# ---------------------------------------------------------------------------


def test_outputs_rewritten_replay_only(
    stores: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    output_dir, ledger_dir = stores
    stored = _stored_frame(output_dir)
    replay_keys = [
        (row.game_id, row.target)
        for row in stored.itertuples()
        if row.provenance == PROVENANCE_BACKTEST_REPLAY
    ]
    log = _Log()
    monkeypatch.setattr(cli, "LOG", log)

    code = cli.main(
        ["--apply", "--output-dir", str(output_dir), "--ledger-dir", str(ledger_dir)]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "MIGRATE_ROWS= 6" in out
    assert "OUTPUTS_REPLAY_ROWS= 3" in out
    assert "NEXT=" in out
    rewritten = read_bet_list_artifact(output_dir)
    assert set(rewritten["provenance"]) == {PROVENANCE_BACKTEST_REPLAY}
    assert (
        list(zip(rewritten["game_id"], rewritten["target"], strict=True)) == replay_keys
    )

    # The tracker is regenerated from the replay rows: byte-equal to a fresh pair built from them.
    reference = tmp_path / "reference"
    write_bet_list_pair(rewritten, reference)
    assert (output_dir / BET_TRACKER_ARTIFACT_NAME).read_bytes() == (
        reference / BET_TRACKER_ARTIFACT_NAME
    ).read_bytes()
    assert [event for event, _ in log.events] == ["migration"]
    assert log.events[0][1]["rows"] == 6


def test_second_apply_refused(stores: tuple[Path, Path]) -> None:
    output_dir, ledger_dir = stores
    _apply(output_dir, ledger_dir)
    before = {
        path: _sha(path)
        for path in (
            ledger_path(ledger_dir),
            output_dir / BET_LIST_ARTIFACT_NAME,
            output_dir / BET_TRACKER_ARTIFACT_NAME,
        )
    }

    with pytest.raises(AlreadyMigratedError, match="already holds"):
        _apply(output_dir, ledger_dir)

    assert {path: _sha(path) for path in before} == before


def test_backfilled_stamp_refused(stores: tuple[Path, Path]) -> None:
    output_dir, ledger_dir = stores
    payloads = build_migrated_entries(read_bet_list_artifact(output_dir))
    payloads[0].immutable["recipe_id"] = "recipe-2026-row19-v1"

    with pytest.raises(MigrationStampError, match="recipe_id"):
        commit_migrated_rows(ledger_dir, payloads)

    assert not ledger_path(ledger_dir).exists()


def test_finish_completes_interrupted_rewrite(stores: tuple[Path, Path]) -> None:
    output_dir, ledger_dir = stores
    # The interrupted --apply: the ledger is written and verified, the outputs still hold forward.
    commit_migrated_rows(
        ledger_dir, build_migrated_entries(read_bet_list_artifact(output_dir))
    )
    ledger_sha = _sha(ledger_path(ledger_dir))

    report = migrate_forward_rows(output_dir, ledger_dir, finish=True, log=_Log())

    assert report.outputs_rewritten is True
    assert set(read_bet_list_artifact(output_dir)["provenance"]) == {
        PROVENANCE_BACKTEST_REPLAY
    }
    assert _sha(ledger_path(ledger_dir)) == ledger_sha


def test_finish_refuses_a_ledger_that_differs(tmp_path: Path) -> None:
    output_dir, ledger_dir = tmp_path / "outputs", tmp_path / "ledger"
    _write_fixture(output_dir)
    commit_migrated_rows(
        ledger_dir, build_migrated_entries(read_bet_list_artifact(output_dir))
    )
    # The outputs' forward rows no longer equal what the ledger holds.
    changed = _stored_rows()
    changed[4]["per_bet_ev"] = 0.0999
    stored_sha = _sha(_write_fixture(output_dir, changed))

    with pytest.raises(MigrationMismatchError, match="2026_04_ATL_NO"):
        migrate_forward_rows(output_dir, ledger_dir, finish=True, log=_Log())

    assert _sha(output_dir / BET_LIST_ARTIFACT_NAME) == stored_sha


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(stores: tuple[Path, Path], capsys) -> None:
    output_dir, ledger_dir = stores
    before = {
        path.name: _sha(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }

    code = cli.main(["--output-dir", str(output_dir), "--ledger-dir", str(ledger_dir)])

    assert code == 0
    out = capsys.readouterr().out
    assert "MIGRATE_ROWS= 6" in out
    assert "WEEKS= 3,4,5" in out
    assert "STATUS_COUNTS= live=4 suppressed=2" in out
    assert "HEAD_HASH= " in out
    assert "NEXT=" not in out
    assert {
        path.name: _sha(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    } == before
    assert not ledger_dir.exists()
