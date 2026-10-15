"""The forward write exposes the rows it newly appended (Phase 35, Plan 35-04; D-14, SPEC R3).

``forward_ledger.runner.RecordOutcome.appended_rows`` carries the canonical values a
``record_forward_slate`` call wrote, so the bets push and the end-of-run notifier read exactly what
the run wrote and never recompute it. An identical re-run exposes nothing, a re-run adding one row
exposes that row alone, and a late write raises and leaves every byte under ``ledger/`` unchanged.
``pipeline.daily_steps.DailySlate.appended_rows`` collects them across the recommend step's re-runs.

The runner's own test environment is reused, never duplicated: importing ``env`` and
``run_log_path`` registers those fixtures here. Every store lives under ``tmp_path``; nothing reads
or writes the repository's ``ledger/``, ``logs/``, ``data/`` or ``artifacts/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.weekly_bet_list import DecisionBundle, PublishDeadlinePassedError
from forward_ledger import runner
from forward_ledger.store import read_entries
from pipeline import daily_steps, live_skip
from pipeline.daily_steps import DailySlate, slate_lock
from tests.unit.test_forward_ledger_runner import (  # noqa: F401 -- env, run_log_path: fixtures
    DECIDED_AT,
    GAMES,
    LOCK,
    SEASON,
    WEEK,
    _decided_row,
    _Env,
    _ledger_files,
    _sha,
    env,
    run_log_path,
)
from utils.game_lock import ET

EXTRA_GAME = "2026_06_NYJ_MIA"

#: The columns the push and the notifier read from an appended row (D-14).
PUSH_COLUMNS: tuple[str, ...] = (
    "game_id",
    "target",
    "arm",
    "status",
    "per_bet_ev",
    "stake_units",
    "ev_tier",
    "line",
    "bet_side",
)


def _with_extra_game(ledger_env: _Env) -> Any:
    """A decision seam that adds :data:`EXTRA_GAME` to the environment's two-game slate."""

    def _builder(season: int, week: int, **kwargs: Any) -> DecisionBundle:
        bundle = ledger_env.builder(season, week, **kwargs)
        extra = pd.DataFrame([_decided_row(EXTRA_GAME, kwargs["now"])])
        return DecisionBundle(
            frame=pd.concat([bundle.frame, extra], ignore_index=True),
            candidates=bundle.candidates,
            schedule=bundle.schedule,
            gold_inputs=bundle.gold_inputs,
            fits=bundle.fits,
            resolved=bundle.resolved,
            chain_fit_path=bundle.chain_fit_path,
            run_instant=bundle.run_instant,
        )

    return _builder


def _ledger_digests(ledger_dir: Path) -> dict[str, str]:
    """The SHA-256 of every file under *ledger_dir*: rows, snapshots and side copies."""
    return {name: _sha(ledger_dir / name) for name in _ledger_files(ledger_dir)}


# ---------------------------------------------------------------------------
# record_forward_slate
# ---------------------------------------------------------------------------


def test_a_write_exposes_exactly_the_appended_rows(env: _Env) -> None:  # noqa: F811
    outcome = env.record(bundle_builder=_with_extra_game(env))

    assert outcome.appended == 3
    assert len(outcome.appended_rows) == 3
    for row in outcome.appended_rows:
        assert set(PUSH_COLUMNS) <= set(row), sorted(set(PUSH_COLUMNS) - set(row))
    stored = [entry.immutable for entry in read_entries(env.ledger)]
    assert list(outcome.appended_rows) == stored, (
        "the exposed rows are not the values the ledger stored"
    )


def test_an_identical_rerun_exposes_nothing(env: _Env) -> None:  # noqa: F811
    env.record()

    outcome = env.record(decided_at=DECIDED_AT + timedelta(minutes=12))

    assert outcome.appended == 0
    assert outcome.appended_rows == ()


def test_a_rerun_adding_one_row_exposes_only_that_row(env: _Env) -> None:  # noqa: F811
    env.record()

    outcome = env.record(
        decided_at=DECIDED_AT + timedelta(minutes=5),
        bundle_builder=_with_extra_game(env),
    )

    assert outcome.appended == 1
    (row,) = outcome.appended_rows
    assert row["game_id"] == EXTRA_GAME
    assert row == read_entries(env.ledger)[-1].immutable


def test_a_late_write_raises_and_writes_nothing(env: _Env) -> None:  # noqa: F811
    env.record()
    before = _ledger_digests(env.ledger)
    assert before, "the first write stored nothing, so nothing was tested"

    with pytest.raises(PublishDeadlinePassedError, match="after its publish deadline"):
        env.record(
            decided_at=DECIDED_AT + timedelta(minutes=5),
            bundle_builder=_with_extra_game(env),
            publish_by=LOCK,
            clock=lambda: LOCK + timedelta(seconds=1),
        )

    assert _ledger_digests(env.ledger) == before


# ---------------------------------------------------------------------------
# recommend_slate
# ---------------------------------------------------------------------------


def _future_slate() -> DailySlate:
    """The two runner games, kicking off 30 days from now so no lock has passed."""
    kickoff = datetime.combine(
        datetime.now(UTC).date() + timedelta(days=30), time(17), tzinfo=UTC
    )
    run_date = kickoff.astimezone(ET).date() - timedelta(days=1)
    schedule = pd.DataFrame(
        {
            "game_id": list(GAMES),
            "season": [SEASON] * len(GAMES),
            "week": [WEEK] * len(GAMES),
            "kickoff_et": [kickoff] * len(GAMES),
        }
    )
    return DailySlate(
        run_date_et=run_date, lock=slate_lock(run_date), schedule=schedule
    )


def _outcome(*game_ids: str) -> runner.RecordOutcome:
    rows = tuple({"game_id": game_id, "target": "ats"} for game_id in game_ids)
    return runner.RecordOutcome(
        decided=len(rows),
        appended=len(rows),
        identical=0,
        refused=(),
        wrote=True,
        head_hash="h",
        entry_count=len(rows),
        appended_rows=rows,
    )


def test_recommend_slate_accumulates_appended_rows_across_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipeline.steps as steps_mod
    from forward_ledger import cutover

    slate = _future_slate()
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)
    monkeypatch.setattr(steps_mod, "_week_schedule", lambda _s, _w: slate.schedule)
    live_skip.reset_excluded_games()
    outcomes = iter((_outcome(GAMES[0], GAMES[1]), _outcome(EXTRA_GAME)))
    monkeypatch.setattr(
        runner, "record_forward_slate", lambda _slate, **_kwargs: next(outcomes)
    )

    daily_steps.recommend_slate(slate)
    daily_steps.recommend_slate(slate)

    assert [row["game_id"] for row in slate.appended_rows] == [*GAMES, EXTRA_GAME]
