"""The closing transform kind and the zero-credit credit read (Phase 34, LDGR-07, Plan 34-14).

THE TWO KINDS OF CAPTURE
------------------------
The decision capture runs before a game's lock and REFUSES every capture observed after it
(33.2 review C1 CR-02): a row is stamped ``snapshot_ts = lock``, which every lock fence reads as
"known at the lock". A closing capture is post-lock by definition, so it gets its own kind,
``capture_kind="closing"``: it keeps ``snapshot_ts = lock`` (the column's documented meaning) and
the TRUE ``created_at``, and refuses instead any capture at or after the game's SCHEDULED kickoff
-- an in-play line is never a closing line. The scheduled kickoff is the matched schedule row's
``kickoff_et``, derived before either kind's check; a matched row with no kickoff is refused by
name, never judged against a guessed one.

Readers stay correct because they judge admissibility on ``created_at``
(``tests/unit/test_closing_rows_never_priced.py``). The decision path never passes the closing
kind (the AST test below).

THE CREDIT READ
---------------
``OddsAPIClient.remaining_credits`` reads ``x-requests-remaining`` from the FREE ``/v4/sports``
endpoint (0 credits), never from the odds endpoint, and never logs the key.

EVERYTHING HERE IS OFFLINE: the API client is a stand-in or an httpx MockTransport.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

import scripts.ingest_odds as ingest_odds_module
from scripts.ingest_odds import LiveOddsMatchReport, OddsAPIClient, OddsDataIngester
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]

GAME_ID = "2026_W06_DAL@PHI"
# Sunday 1:00 PM ET. Its lock is Saturday 18:00 ET.
KICKOFF = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
LOCK = game_lock(KICKOFF)
DECISION_CAPTURE = datetime(
    2026, 10, 10, 21, 0, tzinfo=UTC
)  # 17:00 ET Saturday, pre-lock
CLOSING_CAPTURE = KICKOFF - timedelta(minutes=30)


def _schedule(kickoff: Any = KICKOFF) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": GAME_ID,
                "season": 2026,
                "week": 6,
                "home_team": "PHI",
                "away_team": "DAL",
                "kickoff_et": kickoff,
            }
        ]
    )


def _event(commence_time: str = "2026-10-11T17:00:00Z") -> dict[str, Any]:
    """One Odds API v4 event: Philadelphia -3.5 at home, total 44.5."""
    return {
        "id": "event0",
        "commence_time": commence_time,
        "home_team": "Philadelphia Eagles",
        "away_team": "Dallas Cowboys",
        "bookmakers": [
            {
                "key": "draftkings",
                "last_update": "2026-10-11T16:20:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Philadelphia Eagles", "price": -150},
                            {"name": "Dallas Cowboys", "price": 130},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {
                                "name": "Philadelphia Eagles",
                                "price": -110,
                                "point": -3.5,
                            },
                            {"name": "Dallas Cowboys", "price": -110, "point": 3.5},
                        ],
                    },
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


def _bare_ingester() -> OddsDataIngester:
    """The transformer alone: no settings, no database, no API client."""
    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.sportsbook_priority = []
    return ingester


def _transform(
    captured_at: datetime,
    *,
    schedule: pd.DataFrame | None = None,
    event: dict[str, Any] | None = None,
    **kind: Any,
) -> tuple[pd.DataFrame, LiveOddsMatchReport]:
    ingester = _bare_ingester()
    frame = ingester.transform_odds_data(
        [event or _event()],
        schedule=_schedule() if schedule is None else schedule,
        locks={GAME_ID: LOCK},
        captured_at=captured_at,
        **kind,
    )
    return frame, ingester.last_match_report


# ---------------------------------------------------------------------------
# The closing kind
# ---------------------------------------------------------------------------


class TestTheClosingKind:
    """Post-lock, pre-kickoff captures are written with their true capture instant."""

    def test_closing_kind_keeps_post_lock_rows(self) -> None:
        assert CLOSING_CAPTURE > LOCK, "the fixture capture must be post-lock"
        frame, report = _transform(CLOSING_CAPTURE, capture_kind="closing")

        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["snapshot_ts"] == LOCK, (
            "snapshot_ts keeps its meaning: the game's lock"
        )
        assert row["created_at"] == CLOSING_CAPTURE, (
            "created_at is the true capture instant"
        )
        assert bool(row["is_live"]) is False
        assert row["spread"] == 3.5
        assert row["total"] == 44.5
        assert report.post_lock_games == ()
        assert report.in_play_games == ()

    def test_closing_rows_pass_the_schema(self) -> None:
        """The validated frame is what the store receives; nothing is dropped on the way."""
        ingester = _bare_ingester()
        frame = ingester.transform_odds_data(
            [_event()],
            schedule=_schedule(),
            locks={GAME_ID: LOCK},
            captured_at=CLOSING_CAPTURE,
            capture_kind="closing",
        )
        validated = ingester.validate_odds_data(frame)
        assert len(validated) == 1
        assert validated.iloc[0]["created_at"] == CLOSING_CAPTURE

    def test_one_second_before_kickoff_is_written(self) -> None:
        """Boundary control: the refusal below sits ON the kickoff, not before it."""
        frame, _report = _transform(
            KICKOFF - timedelta(seconds=1), capture_kind="closing"
        )
        assert len(frame) == 1

    @pytest.mark.parametrize(
        "captured_at",
        [KICKOFF, KICKOFF + timedelta(seconds=1), KICKOFF + timedelta(hours=1)],
        ids=["at_kickoff", "one_second_after", "in_play"],
    )
    def test_closing_kind_refuses_in_play(self, captured_at: datetime) -> None:
        frame, report = _transform(captured_at, capture_kind="closing")
        assert frame.empty, "an in-play line is never written"
        assert report.in_play_games == (GAME_ID,)

    def test_the_kickoff_is_the_schedule_rows_never_the_payloads(self) -> None:
        """The payload says 3:00 PM; the schedule says 1:00 PM. The schedule decides."""
        frame, report = _transform(
            KICKOFF + timedelta(minutes=30),
            event=_event(commence_time="2026-10-11T19:00:00Z"),
            capture_kind="closing",
        )
        assert frame.empty
        assert report.in_play_games == (GAME_ID,)

    def test_a_null_scheduled_kickoff_is_refused_by_name(self) -> None:
        frame, report = _transform(
            CLOSING_CAPTURE, schedule=_schedule(kickoff=pd.NaT), capture_kind="closing"
        )
        assert frame.empty, "a capture is never judged against a guessed kickoff"
        assert report.no_kickoff_games == (GAME_ID,)
        assert report.in_play_games == ()

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="capture_kind"):
            _transform(CLOSING_CAPTURE, capture_kind="close")


class TestTheDecisionKindIsUnchanged:
    """The default kind still refuses every post-lock capture (33.2 review C1 CR-02)."""

    @pytest.mark.parametrize("kind", [{}, {"capture_kind": "decision"}])
    def test_decision_kind_refuses_post_lock(self, kind: dict[str, str]) -> None:
        frame, report = _transform(CLOSING_CAPTURE, **kind)
        assert frame.empty
        assert report.post_lock_games == (GAME_ID,)
        assert report.in_play_games == ()

    @pytest.mark.parametrize("kind", [{}, {"capture_kind": "decision"}])
    def test_decision_kind_writes_pre_lock(self, kind: dict[str, str]) -> None:
        frame, report = _transform(DECISION_CAPTURE, **kind)
        assert len(frame) == 1
        assert frame.iloc[0]["snapshot_ts"] == LOCK
        assert frame.iloc[0]["created_at"] == DECISION_CAPTURE
        assert report.post_lock_games == ()

    def test_at_lock_is_still_admissible(self) -> None:
        frame, _report = _transform(LOCK)
        assert len(frame) == 1


# ---------------------------------------------------------------------------
# The decision path never passes the closing kind (source scan)
# ---------------------------------------------------------------------------


def _capture_kind_keywords(tree: ast.AST) -> list[ast.keyword]:
    """Every ``capture_kind=`` keyword passed to any call in *tree*."""
    return [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "capture_kind"
    ]


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    msg = f"{class_name}.{method_name} not found"
    raise AssertionError(msg)


class TestTheDecisionPathNeverUsesTheClosingKind:
    """The daily decision capture can only ever run the default (decision) kind."""

    def test_the_scan_sees_a_closing_kind_call(self) -> None:
        """Positive control: the finder is not vacuous."""
        tree = ast.parse("ingester.transform_odds_data(raw, capture_kind='closing')")
        assert len(_capture_kind_keywords(tree)) == 1

    def test_decision_path_never_uses_closing_kind(self) -> None:
        daily_steps = ast.parse(
            (REPO_ROOT / "pipeline" / "daily_steps.py").read_text(encoding="utf-8")
        )
        assert _capture_kind_keywords(daily_steps) == [], (
            "pipeline/daily_steps.py passes capture_kind; the decision capture must use the "
            "default decision kind"
        )

        ingest_module = ast.parse(
            (REPO_ROOT / "scripts" / "ingest_odds.py").read_text(encoding="utf-8")
        )
        ingest = _method(ingest_module, "OddsDataIngester", "ingest_odds")
        assert _capture_kind_keywords(ingest) == [], (
            "OddsDataIngester.ingest_odds passes capture_kind; the decision ingest must use "
            "the default decision kind"
        )


# ---------------------------------------------------------------------------
# The zero-credit credit read
# ---------------------------------------------------------------------------

_SECRET = "test-odds-key-0123456789"


class _RecordingLogger:
    """Records every log call's message and fields."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str, message: str, **fields: Any) -> None:
        self.calls.append((level, message, fields))

    def debug(self, message: str, **fields: Any) -> None:
        self._record("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._record("info", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._record("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._record("error", message, **fields)


def _client(headers: dict[str, str], requests: list[httpx.Request]) -> OddsAPIClient:
    """An OddsAPIClient whose transport answers every request with *headers*. No network."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json=[{"key": "americanfootball_nfl"}], headers=headers
        )

    client = OddsAPIClient.__new__(OddsAPIClient)
    client.api_key = _SECRET
    client.base_url = "https://odds.invalid/v4"
    client.mock_mode = False
    client.mock_season_week = None
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


class TestRemainingCredits:
    """The remaining-credit count comes from the free sports endpoint's header."""

    def test_remaining_credits_reads_header_without_odds_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(ingest_odds_module, "logger", recorder)
        requests: list[httpx.Request] = []
        client = _client({"x-requests-remaining": "494"}, requests)

        assert client.remaining_credits() == 494

        assert [request.url.path for request in requests] == ["/v4/sports"], (
            "the credit read must call only the free sports endpoint, never the odds endpoint"
        )
        logged_params = [
            fields["params"]
            for _level, _msg, fields in recorder.calls
            if "params" in fields
        ]
        assert logged_params, "the request parameters were not logged at all"
        assert all(params["apiKey"] == "***REDACTED***" for params in logged_params)
        assert _SECRET not in repr(recorder.calls), "the API key reached the log"

    @pytest.mark.parametrize(
        "headers",
        [{}, {"x-requests-remaining": "lots"}, {"x-requests-remaining": "-3"}],
        ids=["missing", "garbage", "negative"],
    )
    def test_an_unreadable_header_is_none(self, headers: dict[str, str]) -> None:
        requests: list[httpx.Request] = []
        assert _client(headers, requests).remaining_credits() is None
        assert len(requests) == 1

    def test_mock_mode_reads_nothing(self) -> None:
        requests: list[httpx.Request] = []
        client = _client({"x-requests-remaining": "494"}, requests)
        client.mock_mode = True
        assert client.remaining_credits() is None
        assert requests == []
