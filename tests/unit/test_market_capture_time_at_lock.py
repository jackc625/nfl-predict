"""A market line counts for a game only with a REAL capture time at or before its lock.

Plan 33.2-14, owner ruling 2026-09-22 (Option B, "Only real capture times"), recorded in the
phase's ``deferred-items.md``. The odds table's ``snapshot_ts`` is a LABEL, not a time anything
was known: every stored 2018-2024 row carries one manufactured constant per season (18:00 ET on
September 19, after 210 week-1/2 games were played), 2025 carries the retired preceding-Friday
freeze, and the live capture path writes each game's own lock into it. None of those says when
the line was seen. ``created_at`` does: the live capture path records the instant the response
was observed (``scripts/ingest_odds.py``), and a backfill records the instant it ran.

So the market-anchor builder admits a row for a game only when its ``created_at`` is present and
AT OR BEFORE that game's lock (``utils.game_lock.is_admissible``; at-lock admissible, one second
later not). Consequences pinned here:

* a 2026-shaped row captured before the lock is admitted, and its reported information time is
  its capture time -- never the stamp in ``snapshot_ts``;
* the same row captured one second after the lock is refused, and the game is the honest
  unknown: every market value NULL, ``basis="no_information"``, checked against a non-empty
  signature of NULLs -- never a 0.0 movement or a 0.5 probability stand-in;
* a 2018-2024-shaped row (``created_at`` NULL, Plan 33.2-08 nulled the 1970 family) is never
  admitted, even when its manufactured ``snapshot_ts`` sits days before the lock;
* a 2025-shaped row (``created_at`` the 2026-09-05 backfill) is never admitted for a game its
  backfill post-dates.

Every value here is synthetic and nothing reads or writes ``data/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.market_anchors import MarketAnchorFeaturesCalculator
from features.protocol import InformationTimeProvider
from features.provenance import (
    PROVENANCE_COLUMNS,
    InformationBasis,
    InformationTimeGate,
    SourceCheckState,
)

ET = ZoneInfo("America/New_York")
_ONE_SECOND = timedelta(seconds=1)
_AS_OF = datetime(2030, 1, 1, tzinfo=ET)  # carried for the Protocol only; not a fence

MARKET_COLUMNS = (
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
)


def _game(game_id: str, season: int, week: int, kickoff: datetime) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": kickoff,
            }
        ]
    )
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _odds_row(
    game_id: str,
    *,
    snapshot_ts: str,
    created_at: pd.Timestamp | None,
    spread: float = -3.5,
) -> dict:
    return {
        "game_id": game_id,
        "sportsbook": "consensus",
        "snapshot_ts": snapshot_ts,
        "created_at": created_at,
        "ml_home": -160.0,
        "ml_away": 140.0,
        "spread": spread,
        "total": 45.5,
        "spread_ju_home": -110.0,
        "spread_ju_away": -110.0,
        "total_over_ju": -110.0,
        "total_under_ju": -110.0,
    }


def _build(games: pd.DataFrame, rows: list[dict]):
    odds = pd.DataFrame(rows)
    odds["created_at"] = pd.to_datetime(odds["created_at"], utc=True)
    calculator = MarketAnchorFeaturesCalculator()
    with patch("features.market_anchors.load_dataframe", return_value=odds):
        frame = calculator.build_features(games, _AS_OF)
        provenance = calculator.information_times(games)
    return calculator, frame, provenance


# A 2026 Sunday game: its lock is Saturday 18:00 ET.
_LIVE_ID = "2026_W03_BUF@KC"
_LIVE = _game(_LIVE_ID, 2026, 3, datetime(2026, 9, 27, 13, 0, tzinfo=ET))
_LIVE_LOCK = pd.Timestamp(lock_rule.lock_frame(_LIVE)[_LIVE_ID])


class TestTheMarketBuilderIsAProvenanceSupplier:
    def test_it_satisfies_the_information_time_protocol(self) -> None:
        assert isinstance(MarketAnchorFeaturesCalculator(), InformationTimeProvider)

    def test_the_no_information_signature_is_every_market_value_null(self) -> None:
        signature = dict(MarketAnchorFeaturesCalculator().no_information_signature())
        assert signature == dict.fromkeys(MARKET_COLUMNS)


class TestARealPreLockCaptureIsAdmitted:
    def test_a_2026_capture_before_the_lock_supplies_the_line(self) -> None:
        captured = _LIVE_LOCK - timedelta(hours=6)
        _, frame, provenance = _build(
            _LIVE,
            [_odds_row(_LIVE_ID, snapshot_ts=str(_LIVE_LOCK), created_at=captured)],
        )
        row = frame.iloc[0]
        assert row["snapshot_spread"] == pytest.approx(-3.5)
        assert row["snapshot_total"] == pytest.approx(45.5)
        assert list(provenance.columns) == list(PROVENANCE_COLUMNS)
        assert provenance.iloc[0]["basis"] == InformationBasis.PER_ROW.value
        assert pd.Timestamp(provenance.iloc[0]["information_time"]) == captured

    def test_a_capture_exactly_at_the_lock_is_admitted(self) -> None:
        _, frame, provenance = _build(
            _LIVE,
            [_odds_row(_LIVE_ID, snapshot_ts=str(_LIVE_LOCK), created_at=_LIVE_LOCK)],
        )
        assert frame.iloc[0]["snapshot_spread"] == pytest.approx(-3.5)
        assert pd.Timestamp(provenance.iloc[0]["information_time"]) == _LIVE_LOCK

    def test_the_reported_time_is_the_capture_never_the_snapshot_ts_label(self) -> None:
        """The live path writes the LOCK into snapshot_ts; the capture is what is reported."""
        captured = _LIVE_LOCK - timedelta(days=2)
        _, _, provenance = _build(
            _LIVE,
            [_odds_row(_LIVE_ID, snapshot_ts=str(_LIVE_LOCK), created_at=captured)],
        )
        reported = pd.Timestamp(provenance.iloc[0]["information_time"])
        assert reported == captured
        assert reported != _LIVE_LOCK

    def test_the_freshest_admissible_capture_wins(self) -> None:
        early = _LIVE_LOCK - timedelta(days=3)
        late = _LIVE_LOCK - timedelta(hours=1)
        after = _LIVE_LOCK + timedelta(hours=1)
        _, frame, provenance = _build(
            _LIVE,
            [
                _odds_row(_LIVE_ID, snapshot_ts="x", created_at=early, spread=-2.5),
                _odds_row(_LIVE_ID, snapshot_ts="x", created_at=late, spread=-4.0),
                _odds_row(_LIVE_ID, snapshot_ts="x", created_at=after, spread=-9.0),
            ],
        )
        row = frame.iloc[0]
        assert row["snapshot_spread"] == pytest.approx(-4.0)
        assert row["spread_movement"] == pytest.approx(-1.5)
        assert pd.Timestamp(provenance.iloc[0]["information_time"]) == late


class TestAPostLockCaptureIsRefused:
    def test_one_second_after_the_lock_the_game_is_the_honest_unknown(self) -> None:
        calculator, frame, provenance = _build(
            _LIVE,
            [
                _odds_row(
                    _LIVE_ID,
                    snapshot_ts=str(_LIVE_LOCK),
                    created_at=_LIVE_LOCK + _ONE_SECOND,
                )
            ],
        )
        row = frame.iloc[0]
        for column in MARKET_COLUMNS:
            assert pd.isna(row[column]), f"{column} must be NULL, got {row[column]!r}"
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value
        assert pd.isna(provenance.iloc[0]["information_time"])
        state = InformationTimeGate().check(
            "market",
            frame,
            provenance,
            lock_rule.lock_frame(_LIVE),
            no_information_signature=calculator.no_information_signature(),
        )
        assert state is SourceCheckState.CHECKED


class TestAFabricatedStampIsNeverAnInformationTime:
    def test_a_2018_row_with_only_its_season_stamp_is_never_admitted(self) -> None:
        """2018-2024 shape: snapshot_ts is 18:00 ET Sep 19 -- days BEFORE this week-3
        lock -- and created_at is NULL. The stamp alone admits nothing."""
        game_id = "2018_W03_BUF@KC"
        games = _game(game_id, 2018, 3, datetime(2018, 9, 23, 13, 0, tzinfo=ET))
        lock = pd.Timestamp(lock_rule.lock_frame(games)[game_id])
        stamp = "2018-09-19T18:00:00-04:00"
        assert pd.Timestamp(stamp) < lock  # the label would pass a snapshot_ts fence
        _, frame, provenance = _build(
            games, [_odds_row(game_id, snapshot_ts=stamp, created_at=None)]
        )
        assert frame.iloc[0][list(MARKET_COLUMNS)].isna().all()
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value

    def test_a_2025_row_backfilled_after_the_game_is_never_admitted(self) -> None:
        game_id = "2025_W01_DAL@PHI"
        games = _game(game_id, 2025, 1, datetime(2025, 9, 4, 20, 20, tzinfo=ET))
        backfill = pd.Timestamp("2026-09-05 04:59:49.969295", tz="UTC")
        _, frame, provenance = _build(
            games,
            [
                _odds_row(
                    game_id,
                    snapshot_ts="2025-08-29 22:00:00+00:00",
                    created_at=backfill,
                )
            ],
        )
        assert frame.iloc[0][list(MARKET_COLUMNS)].isna().all()
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value

    def test_a_game_with_no_odds_row_is_the_honest_unknown_too(self) -> None:
        other = _odds_row("2026_W03_NYJ@NE", snapshot_ts="x", created_at=_LIVE_LOCK)
        _, frame, provenance = _build(_LIVE, [other])
        assert frame.iloc[0][list(MARKET_COLUMNS)].isna().all()
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value
