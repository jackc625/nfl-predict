"""The A/A proof and the per-week comparability invariant (Plan 37-08; SPEC R7, R8, D-02, D-03).

Every fixture week is written through the real store (``append_rows`` / ``apply_updates``) into a
tmp ledger and read back with ``read_entries``, so the evaluation sees exactly what production
would. Nothing here touches the production ledger or the network.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from forward_ledger.store import append_rows, apply_updates, read_entries
from refit.aa import evaluate_week, record_week, week_first_decision_run
from refit.records import read_aa_records
from tests.unit.test_forward_ledger_store import graded, key_of, make_row

SEASON = 2026
WEEK = 8
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# Week 8's three fixture games: Thursday night, Sunday after DST ends, Monday night.
WEEK8_KICKOFFS: dict[str, str] = {
    "2026_08_KC_BUF": "2026-10-29T20:15:00-04:00",
    "2026_08_DAL_PHI": "2026-11-01T13:00:00-05:00",
    "2026_08_SF_SEA": "2026-11-02T20:15:00-05:00",
}

# The Thursday game locks Wed 2026-10-28 18:00 EDT (22:00Z); its decision run starts one hour
# earlier. The real pre-registration commit (2026-10-20) precedes it.
FIRST_DECISION_RUN = datetime(2026, 10, 28, 21, 0, tzinfo=UTC)
COMMITTED_BEFORE = datetime(2026, 10, 20, 19, 41, 23, tzinfo=UTC)

# The evaluation instant of week 8: Tue 2026-11-03 17:00 EST.
EVALUATION_INSTANT = "2026-11-03T22:00:00+00:00"


def games_frame(kickoffs: dict[str, str] = WEEK8_KICKOFFS) -> pd.DataFrame:
    """A silver-``games``-shaped week-8 schedule."""
    return pd.DataFrame(
        {
            "game_id": list(kickoffs),
            "season": [SEASON] * len(kickoffs),
            "week": [WEEK] * len(kickoffs),
            "kickoff_et": [pd.Timestamp(value) for value in kickoffs.values()],
        }
    )


def week8_rows(
    arm: str, games: tuple[str, ...] = tuple(WEEK8_KICKOFFS)
) -> list[dict[str, Any]]:
    """One row per game and target of week 8 in *arm*, decided at the first decision run."""
    return [
        make_row(
            game_id=game_id,
            week=WEEK,
            target=target,
            freeze_ts="2026-10-28T18:00:00-04:00",
            decided_at_utc="2026-10-28T21:10:00+00:00",
            arm=arm,
        )
        for game_id in games
        for target in TARGETS
    ]


def write_twins(
    ledger_dir: Path,
    live: list[dict[str, Any]],
    shadow: list[dict[str, Any]],
) -> None:
    """Append both arms and grade every row of both arms a win, as one settle commit would."""
    append_rows(ledger_dir, [*live, *shadow])
    apply_updates(
        ledger_dir,
        grading_updates={
            key_of(row): graded("win", 0.9090909090909091, 1.1363636363636365)
            for row in [*live, *shadow]
        },
    )


def evaluate(ledger_dir: Path, games: pd.DataFrame | None = None, **kwargs: Any):
    return evaluate_week(
        read_entries(ledger_dir),
        games_frame() if games is None else games,
        SEASON,
        WEEK,
        committed_at=kwargs.pop("committed_at", COMMITTED_BEFORE),
    )


# ---------------------------------------------------------------------------
# Task 1: the A/A week (R7, D-02, D-03)
# ---------------------------------------------------------------------------


def test_a_clean_week_passes(tmp_path: Path) -> None:
    write_twins(tmp_path, week8_rows("live"), week8_rows("shadow"))

    result = evaluate(tmp_path)

    assert result.clean is True
    assert result.reason == "clean"
    assert result.first_mismatch is None
    assert (result.live_rows, result.twins) == (9, 9)
    assert result.week_first_decision_run_utc == FIRST_DECISION_RUN

    record_week(tmp_path, result, code_commit="abc123")
    records = read_aa_records(tmp_path)
    assert len(records) == 1
    assert (records[0]["season"], records[0]["week"], records[0]["clean"]) == (
        2026,
        8,
        True,
    )
    assert records[0]["reason"] == "clean"
    assert records[0]["code_commit"] == "abc123"
    assert records[0]["first_mismatch"] is None
    assert records[0]["week_first_decision_run_utc"] == FIRST_DECISION_RUN.isoformat()


def test_week_first_decision_run_is_the_first_lock_minus_the_lead() -> None:
    assert week_first_decision_run(games_frame(), SEASON, WEEK) == FIRST_DECISION_RUN


def test_a_planted_field_mismatch_names_the_field(tmp_path: Path) -> None:
    shadow = week8_rows("shadow")
    shadow[4] = {**shadow[4], "per_bet_ev": 0.0378}
    write_twins(tmp_path, week8_rows("live"), shadow)

    result = evaluate(tmp_path)

    assert result.clean is False
    assert result.reason == "field_mismatch"
    assert result.first_mismatch == {
        "game_id": shadow[4]["game_id"],
        "target": shadow[4]["target"],
        "field": "per_bet_ev",
    }


def test_negative_zero_differs_from_zero(tmp_path: Path) -> None:
    live = week8_rows("live")
    shadow = week8_rows("shadow")
    live[0] = {**live[0], "flat_stake": 0.0}
    shadow[0] = {**shadow[0], "flat_stake": -0.0}
    write_twins(tmp_path, live, shadow)

    result = evaluate(tmp_path)

    assert result.reason == "field_mismatch"
    assert result.first_mismatch is not None
    assert result.first_mismatch["field"] == "flat_stake"


def test_a_grading_difference_is_a_mismatch(tmp_path: Path) -> None:
    live = week8_rows("live")
    shadow = week8_rows("shadow")
    append_rows(tmp_path, [*live, *shadow])
    updates = {
        key_of(row): graded("win", 0.9090909090909091, 1.1363636363636365)
        for row in [*live, *shadow]
    }
    # One shadow twin settled a loss where its live twin settled a win.
    updates[key_of(shadow[2])] = graded("loss", -1.0, -1.25)
    apply_updates(tmp_path, grading_updates=updates)

    result = evaluate(tmp_path)

    assert result.clean is False
    assert result.reason == "field_mismatch"
    assert result.first_mismatch == {
        "game_id": shadow[2]["game_id"],
        "target": shadow[2]["target"],
        "field": "grading.grading_status",
    }


def test_two_pending_twins_agree(tmp_path: Path) -> None:
    append_rows(tmp_path, [*week8_rows("live"), *week8_rows("shadow")])

    assert evaluate(tmp_path).clean is True


def test_fill_and_position_fields_are_ignored(tmp_path: Path) -> None:
    live = week8_rows("live")
    write_twins(tmp_path, live, week8_rows("shadow"))
    apply_updates(
        tmp_path,
        fill_updates={
            key_of(live[0]): {
                "fill_sportsbook": "draftkings",
                "fill_line": -3.0,
                "fill_odds": -110.0,
                "fill_stake_dollars": 25.0,
                "fill_at_utc": "2026-10-28T21:30:00+00:00",
            }
        },
    )

    result = evaluate(tmp_path)

    assert result.clean is True
    assert result.reason == "clean"


def test_zero_live_rows_does_not_qualify(tmp_path: Path) -> None:
    result = evaluate(tmp_path)

    assert result.clean is False
    assert result.reason == "no_live_rows"
    assert result.live_rows == 0


def test_missed_runs_do_not_void(tmp_path: Path) -> None:
    played = ("2026_08_KC_BUF", "2026_08_DAL_PHI")
    write_twins(tmp_path, week8_rows("live", played), week8_rows("shadow", played))

    result = evaluate(tmp_path)

    assert result.clean is True
    assert (result.live_rows, result.twins) == (6, 6)


def test_a_missing_twin_is_a_hole(tmp_path: Path) -> None:
    live = week8_rows("live")
    shadow = week8_rows("shadow")
    missing = shadow.pop(5)
    write_twins(tmp_path, live, shadow)

    result = evaluate(tmp_path)

    assert result.clean is False
    assert result.reason == "missing_twin"
    assert result.first_mismatch == {
        "game_id": missing["game_id"],
        "target": missing["target"],
        "field": "twin",
    }


def test_a_shadow_row_without_a_live_twin_is_an_orphan(tmp_path: Path) -> None:
    live = week8_rows("live")
    orphan = live.pop()
    write_twins(tmp_path, live, week8_rows("shadow"))

    result = evaluate(tmp_path)

    assert result.clean is False
    assert result.reason == "orphan_shadow"
    assert result.first_mismatch == {
        "game_id": orphan["game_id"],
        "target": orphan["target"],
        "field": "twin",
    }


def test_preregistration_after_first_run(tmp_path: Path) -> None:
    write_twins(tmp_path, week8_rows("live"), week8_rows("shadow"))

    at_run = evaluate(tmp_path, committed_at=FIRST_DECISION_RUN)
    after_run = evaluate(tmp_path, committed_at=FIRST_DECISION_RUN + timedelta(days=1))
    just_before = evaluate(
        tmp_path, committed_at=FIRST_DECISION_RUN - timedelta(seconds=1)
    )

    for result in (at_run, after_run):
        assert result.clean is False
        assert result.reason == "preregistration_after_first_run"
        assert result.first_mismatch is None
    assert just_before.clean is True
    assert just_before.preregistration_committed_at == FIRST_DECISION_RUN - timedelta(
        seconds=1
    )


def test_a_week_with_a_game_after_its_evaluation_instant_is_not_complete(
    tmp_path: Path,
) -> None:
    postponed = "2026_08_NYJ_MIA"
    at_instant = {**WEEK8_KICKOFFS, postponed: EVALUATION_INSTANT}
    write_twins(tmp_path, week8_rows("live"), week8_rows("shadow"))

    result = evaluate(tmp_path, games_frame(at_instant))

    assert result.clean is False
    assert result.reason == "week_not_complete"
    assert result.first_mismatch is None

    # The same game one second earlier, its twins present: the week is complete and clean.
    earlier = tmp_path / "earlier"
    one_second_before = {**WEEK8_KICKOFFS, postponed: "2026-11-03T21:59:59+00:00"}
    write_twins(
        earlier,
        week8_rows("live", tuple(one_second_before)),
        week8_rows("shadow", tuple(one_second_before)),
    )

    complete = evaluate(earlier, games_frame(one_second_before))

    assert complete.clean is True
    assert complete.reason == "clean"
    assert complete.live_rows == 12
