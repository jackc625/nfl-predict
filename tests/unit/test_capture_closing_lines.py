"""The argument-free closing capture entry point (Phase 34, LDGR-07, Plan 34-14).

``scripts/capture_closing_lines.py`` is what the ``NFL_Predict_Closing`` wake task runs
(``deployment/windows_closing_scheduler.xml``). Each run:

* finds the unplayed games kicking off in ``(now, now + 60:00]`` -- none -> no request, no credit;
* reads the remaining credits from the FREE sports endpoint and asks
  ``forward_ledger.credits.closing_capture_allowed`` -- refused -> skipped with the reason;
* otherwise requests one board bounded by those kickoffs, transforms it with the CLOSING kind
  and appends it to the odds store through the locked writer;
* records exactly one ``closing_capture`` run-log event per outcome (``no_games``, ``skipped``,
  ``captured``, ``failed``), which the settle pass reads for NULL reasons.

The decision capture never consults the credit rule (last test).

EVERYTHING HERE IS OFFLINE: the client is a stand-in, the odds store is sandboxed under
``tmp_path`` and the run log is a recorder. No Odds API call, no credit spent.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data.storage as storage_mod
import scripts.capture_closing_lines as capture_module
import scripts.ingest_odds as ingest_odds_module
from data.storage import ParquetManager
from scripts.ingest_odds import OddsDataIngester
from utils import ExternalAPIError
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sunday 2026-10-11, 12:30 PM ET: the 1:00 PM and 1:25 PM kickoffs are in the next 60 minutes.
NOW = datetime(2026, 10, 11, 16, 30, tzinfo=UTC)
EARLY_KICKOFF = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
LATE_KICKOFF = datetime(2026, 10, 11, 17, 25, tzinfo=UTC)
CAPTURED_AT = NOW + timedelta(seconds=40)

_SECRET = "test-odds-key-0123456789"

_TEAMS: dict[str, tuple[str, str]] = {
    "PHI": ("Philadelphia Eagles", "PHI"),
    "DAL": ("Dallas Cowboys", "DAL"),
    "WAS": ("Washington Commanders", "WAS"),
    "NYG": ("New York Giants", "NYG"),
    "KC": ("Kansas City Chiefs", "KC"),
    "BUF": ("Buffalo Bills", "BUF"),
    "SF": ("San Francisco 49ers", "SF"),
    "SEA": ("Seattle Seahawks", "SEA"),
}


def _game(
    away: str,
    home: str,
    kickoff: datetime,
    *,
    week: int = 6,
    home_score: float | None = None,
) -> dict[str, Any]:
    return {
        "game_id": f"2026_W{week:02d}_{away}@{home}",
        "season": 2026,
        "week": week,
        "home_team": home,
        "away_team": away,
        "kickoff_et": kickoff,
        "home_score": home_score,
        "away_score": home_score,
    }


def _games(*extra: dict[str, Any]) -> pd.DataFrame:
    """Two in-window games, a later-week game each Sunday left in October, and a played game."""
    rows = [
        _game("DAL", "PHI", EARLY_KICKOFF),
        _game("NYG", "WAS", LATE_KICKOFF),
        _game("BUF", "KC", datetime(2026, 10, 18, 17, 0, tzinfo=UTC), week=7),
        _game("SEA", "SF", datetime(2026, 10, 25, 17, 0, tzinfo=UTC), week=8),
        _game(
            "KC", "BUF", datetime(2026, 10, 4, 17, 0, tzinfo=UTC), week=5, home_score=24
        ),
        *extra,
    ]
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _event(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"event-{game['game_id']}",
        "commence_time": game["kickoff_et"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": _TEAMS[game["home_team"]][0],
        "away_team": _TEAMS[game["away_team"]][0],
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 44.5},
                            {"name": "Under", "price": -110, "point": 44.5},
                        ],
                    },
                ],
            }
        ],
    }


class _FakeClient:
    """Stands in for OddsAPIClient. Never touches the network."""

    def __init__(
        self,
        *,
        remaining: int | None = 400,
        payload: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.remaining = remaining
        self.payload = payload or []
        self.error = error
        self.credit_reads = 0
        self.odds_requests: list[dict[str, Any]] = []
        self.closed = False

    def remaining_credits(self) -> int | None:
        self.credit_reads += 1
        return self.remaining

    def get_nfl_odds(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.odds_requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.payload

    def close(self) -> None:
        self.closed = True


class _Run:
    """One sandboxed run of the entry point: its exit code, events, client and store."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        games: pd.DataFrame,
        client: _FakeClient,
    ) -> None:
        self.client = client
        self.events: list[dict[str, Any]] = []
        self.ingesters_made = 0
        self.store = tmp_path / "silver" / "odds_snapshot.parquet"
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )
        monkeypatch.setattr(capture_module, "LOAD_GAMES", games.copy)
        monkeypatch.setattr(capture_module, "CLOCK", self._clock)
        monkeypatch.setattr(capture_module, "MAKE_INGESTER", self._make_ingester)
        monkeypatch.setattr(capture_module, "LOG", self._log)
        self._instants = iter([NOW, CAPTURED_AT])

    def _clock(self) -> datetime:
        return next(self._instants, CAPTURED_AT)

    def _make_ingester(self) -> OddsDataIngester:
        self.ingesters_made += 1
        ingester = OddsDataIngester.__new__(OddsDataIngester)
        ingester.api_client = self.client
        ingester.sportsbook_priority = []
        return ingester

    def _log(self, event: str, **fields: Any) -> bool:
        self.events.append({"event": event, **fields})
        return True

    def main(self, argv: list[str] | None = None) -> int:
        return capture_module.main(argv or [])

    @property
    def event(self) -> dict[str, Any]:
        assert len(self.events) == 1, f"expected one run-log event, got {self.events}"
        assert self.events[0]["event"] == "closing_capture"
        return self.events[0]


def _payload(games: pd.DataFrame) -> list[dict[str, Any]]:
    return [_event(row) for row in games.to_dict("records")]


def _imported_modules(relative: str) -> set[str]:
    """Every module *relative* imports; ``from pkg import mod`` on a bare package is ``pkg.mod``."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "." in node.module:
                imported.add(node.module)
            else:
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


class TestNoGameNear:
    def test_no_games_no_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        games = _games()
        quiet = games.loc[
            ~games["game_id"].isin(["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"])
        ]
        run = _Run(tmp_path, monkeypatch, quiet, _FakeClient())

        assert run.main() == 0
        assert run.ingesters_made == 0, "no game near: no client is even built"
        assert run.client.credit_reads == 0
        assert run.client.odds_requests == []
        assert run.event["outcome"] == "no_games"
        assert run.event["game_ids"] == []
        assert not run.store.exists()


class TestTheCreditRule:
    def test_low_credits_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two decision days remain in October (the 17th and the 24th): reserve 3 x 3 x 2 = 18,
        # and 20 - 3 = 17 falls below it.
        run = _Run(tmp_path, monkeypatch, _games(), _FakeClient(remaining=20))

        assert run.main() == 0
        assert run.client.credit_reads == 1
        assert run.client.odds_requests == [], "a skipped capture makes no odds request"
        assert run.event["outcome"] == "skipped"
        assert run.event["reason"] == "credit_reserve"
        assert run.event["game_ids"] == ["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"]
        assert run.event["remaining_credits"] == 20
        assert not run.store.exists()

    def test_unreadable_header_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _Run(tmp_path, monkeypatch, _games(), _FakeClient(remaining=None))

        assert run.main() == 0
        assert run.client.odds_requests == []
        assert run.event["outcome"] == "skipped"
        assert run.event["reason"] == "credit_header_unreadable"
        assert run.event["game_ids"] == ["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"]


class TestTheCapture:
    def test_capture_writes_closing_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        games = _games()
        in_window = games.loc[
            games["game_id"].isin(["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"])
        ]
        run = _Run(
            tmp_path, monkeypatch, games, _FakeClient(payload=_payload(in_window))
        )

        assert run.main() == 0

        assert len(run.client.odds_requests) == 1, "one board, one request"
        request = run.client.odds_requests[0]
        assert request["date_from"] == EARLY_KICKOFF - timedelta(minutes=1)
        assert request["date_to"] == LATE_KICKOFF + timedelta(minutes=1)

        stored = pd.read_parquet(run.store)
        assert sorted(stored["game_id"]) == ["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"]
        assert set(pd.to_datetime(stored["created_at"], utc=True)) == {
            pd.Timestamp(CAPTURED_AT)
        }
        locks = dict(
            zip(stored["game_id"], pd.to_datetime(stored["snapshot_ts"], utc=True))
        )
        assert locks["2026_W06_DAL@PHI"] == pd.Timestamp(game_lock(EARLY_KICKOFF))
        assert run.client.closed

        event = run.event
        assert event["outcome"] == "captured"
        assert event["game_ids"] == ["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"]
        assert event["rows"] == 2
        assert event["captured_at"] == CAPTURED_AT.isoformat()

    def test_started_games_excluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kicked off five minutes ago, no result recorded yet: never requested.
        started = _game("SEA", "SF", NOW - timedelta(minutes=5))
        games = _games(started)
        in_window = games.loc[
            games["game_id"].isin(["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"])
        ]
        run = _Run(
            tmp_path, monkeypatch, games, _FakeClient(payload=_payload(in_window))
        )

        assert run.main() == 0
        assert run.client.odds_requests[0]["date_from"] == EARLY_KICKOFF - timedelta(
            minutes=1
        )
        assert started["game_id"] not in run.event["game_ids"]

    def test_failure_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        error = ExternalAPIError(f"Request error: refused for apiKey={_SECRET}")
        run = _Run(tmp_path, monkeypatch, _games(), _FakeClient(error=error))

        assert run.main() == 1
        event = run.event
        assert event["outcome"] == "failed"
        assert event["reason"] == "capture_failed"
        assert event["error_type"] == "ExternalAPIError"
        assert event["game_ids"] == ["2026_W06_DAL@PHI", "2026_W06_NYG@WAS"]
        captured = capsys.readouterr()
        assert _SECRET not in repr(run.events)
        assert _SECRET not in captured.out + captured.err
        assert not run.store.exists()

    def test_now_must_carry_an_offset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _Run(tmp_path, monkeypatch, _games(), _FakeClient())
        with pytest.raises(SystemExit) as refused:
            run.main(["--now", "2026-10-11T12:30:00"])
        assert refused.value.code == 2
        assert run.events == []

    def test_now_selects_the_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--now`` (tests only) moves the window; at 1:01 PM ET only the 1:25 PM game is near."""
        games = _games()
        run = _Run(tmp_path, monkeypatch, games, _FakeClient(payload=[]))
        assert run.main(["--now", "2026-10-11T13:01:00-04:00"]) == 0
        assert run.event["game_ids"] == ["2026_W06_NYG@WAS"]

    def test_help_says_no_argument_is_required(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as shown:
            capture_module.main(["--help"])
        assert shown.value.code == 0
        # argparse wraps to the terminal width, so compare with whitespace collapsed.
        assert "no required arguments" in " ".join(capsys.readouterr().out.split())


class TestTheDecisionCaptureIgnoresTheCreditRule:
    """The daily decision capture never consults the closing reserve (D-08)."""

    def test_decision_capture_ignores_credit_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pipeline.daily_steps import DailySlate, capture_slate_odds

        games = _games()
        slate_games = games.loc[games["game_id"] == "2026_W06_DAL@PHI"]
        schedule = slate_games[list(ingest_odds_module.SCHEDULE_COLUMNS)].reset_index(
            drop=True
        )
        client = _FakeClient(remaining=0, payload=_payload(slate_games))

        def _ingester() -> OddsDataIngester:
            ingester = OddsDataIngester.__new__(OddsDataIngester)
            ingester.api_client = client
            ingester.sportsbook_priority = []
            return ingester

        bronze_writes: list[str] = []
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )
        monkeypatch.setattr(
            ingest_odds_module, "load_schedule_slice", lambda *_: schedule
        )
        monkeypatch.setattr(ingest_odds_module, "OddsDataIngester", _ingester)
        monkeypatch.setattr(
            ingest_odds_module,
            "save_dataframe",
            lambda _frame, name, *_a, **_k: bronze_writes.append(name),
        )
        # 17:00 ET on the Saturday: before the game's lock.
        monkeypatch.setattr(
            ingest_odds_module,
            "_observe_capture_instant",
            lambda: datetime(2026, 10, 10, 21, 0, tzinfo=UTC),
        )
        slate = DailySlate(
            run_date_et=date(2026, 10, 10),
            lock=game_lock(EARLY_KICKOFF),
            schedule=schedule,
        )

        capture_slate_odds(slate)

        assert len(client.odds_requests) == 1, (
            "the decision capture still requested its board"
        )
        assert client.credit_reads == 0, (
            "the decision capture must not read the credit rule"
        )
        assert slate.odds_failure is None
        assert len(pd.read_parquet(tmp_path / "silver" / "odds_snapshot.parquet")) == 1

    def test_the_decision_modules_do_not_import_the_credit_rule(self) -> None:
        # Plan 34-15 wires the recommend step to the ledger behind the cutover switch; those two
        # modules are the only ledger imports the decision path may carry, and neither reaches
        # the credit rule.
        allowed = {"forward_ledger.cutover", "forward_ledger.runner"}
        for relative in (
            "pipeline/daily_steps.py",
            "scripts/ingest_odds.py",
            "forward_ledger/runner.py",
        ):
            imported = _imported_modules(relative)
            assert "forward_ledger.credits" not in imported, relative
            if relative != "forward_ledger/runner.py":
                ledger = {
                    name for name in imported if name.startswith("forward_ledger")
                }
                assert ledger <= allowed, (relative, sorted(ledger - allowed))
