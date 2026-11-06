"""The early-lines capture (owner ruling 2026-11-05): one board a day, never at the others' expense.

* With enough credits it requests the whole board (no kickoff window) and appends one JSON line
  holding the capture instant, the credits read before it and the raw board.
* It spends a board only when the month's decision captures (at their retry bound) and closing
  readings stay covered; one credit short, or an unreadable header, makes no request and writes
  nothing.

Every test writes only under ``tmp_path`` and uses a stand-in client: no request leaves the machine.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

import scripts.capture_early_lines as early

NOW = datetime(2026, 11, 5, 22, 5, tzinfo=UTC)  # Thursday 17:05 ET
BOARD = [
    {"id": "e1", "home_team": "Baltimore Ravens", "away_team": "Jacksonville Jaguars"},
    {"id": "e2", "home_team": "Denver Broncos", "away_team": "Kansas City Chiefs"},
]


def _games() -> pd.DataFrame:
    """Two November games left: a Sunday 13:00 ET and a Monday 20:15 ET kickoff.

    Their locks fall on Saturday and Sunday (two decision days), and each has its own closing
    window, so the reserve is 3 x (3 x 2 + 2) = 24 credits and a board, at its retry bound of 9
    credits, needs 33 on hand.
    """
    return pd.DataFrame(
        {
            "game_id": ["2026_W10_KC@DEN", "2026_W10_NYG@DAL"],
            "season": 2026,
            "kickoff_et": [
                pd.Timestamp("2026-11-08 18:00", tz="UTC"),
                pd.Timestamp("2026-11-10 01:15", tz="UTC"),
            ],
        }
    )


class _Client:
    """A stand-in Odds API client that records what was asked of it."""

    def __init__(self, remaining: int | None) -> None:
        self.remaining = remaining
        self.board_requests: list[dict[str, Any]] = []
        self.closed = False

    def remaining_credits(self) -> int | None:
        return self.remaining

    def get_nfl_odds(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.board_requests.append(kwargs)
        return BOARD

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def lines_path(monkeypatch, tmp_path):
    path = tmp_path / "early_lines.jsonl"
    monkeypatch.setattr(early, "EARLY_LINES_PATH", path)
    return path


def _capture(client: _Client):
    return early.capture_early_lines(
        _games(), make_client=lambda: client, clock=lambda: NOW
    )


def test_a_covered_month_records_the_whole_board(lines_path, capsys):
    client = _Client(remaining=33)

    outcome = _capture(client)

    assert outcome == early.EarlyLinesOutcome("captured", None, 2, 33)
    assert client.board_requests == [{}], "the board must be requested with no window"
    assert client.closed
    (line,) = lines_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["captured_at_utc"] == NOW.isoformat()
    assert record["remaining_credits_before"] == 33
    assert record["events"] == 2
    assert record["board"] == BOARD
    assert "EARLY_LINES= captured events=2" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("remaining", "reason"),
    [(32, "credit_reserve"), (None, "credit_header_unreadable")],
)
def test_no_board_is_spent_when_the_other_captures_are_not_covered(
    lines_path, capsys, remaining, reason
):
    client = _Client(remaining=remaining)

    outcome = _capture(client)

    assert outcome == early.EarlyLinesOutcome("skipped", reason, 0, remaining)
    assert client.board_requests == []
    assert client.closed
    assert not lines_path.exists()
    assert f"EARLY_LINES= skipped reason={reason} events=0" in capsys.readouterr().out
