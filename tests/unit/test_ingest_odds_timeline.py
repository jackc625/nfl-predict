"""Mock/stub-mode tests for scripts/ingest_odds_timeline.py (SIG-04).

These tests NEVER make a live (paid) historical call -- the real paid pull is
Plan 29-05 under a paid key. They prove:

1. The backfill loop stamps ``snapshot_ts`` from the ENVELOPE timestamp, NOT the
   requested date T, and the composite-key write is idempotent on a double run
   (review 29-03 HIGH).
2. The Friday-18:00 freeze fence resolves to an ET-evening instant (22:00/23:00
   UTC), never a UTC-localized 18:00 (WR-02).
3. The backfill path HARD-FAILS on a mock-mode client (OUM-06).
4. The dedicated raw-envelope normalizer turns a raw API game envelope into
   consensus ``OddsTimelineSchema`` rows (review 29-03 MED).
5. ``_make_request``'s DEBUG params log redacts ``apiKey`` (review 29-03 MED).
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.ingest_odds import OddsAPIClient
from scripts.ingest_odds_timeline import (
    MockModeBackfillError,
    backfill_timeline,
    normalize_envelope_to_timeline_rows,
    weekly_snapshot_timestamps,
)

ET = ZoneInfo("America/New_York")

# The envelope's actual snapshot timestamp -- deliberately distinct from any of
# the requested weekly cadence timestamps T (review 29-03 HIGH).
ENVELOPE_TS = "2021-10-15T21:57:00Z"


def _fake_envelope(timestamp: str = ENVELOPE_TS) -> dict:
    """A raw historical envelope with one game and two US books (totals)."""
    return {
        "timestamp": timestamp,
        "previous_timestamp": "2021-10-15T21:52:00Z",
        "next_timestamp": "2021-10-15T22:02:00Z",
        "data": [
            {
                "id": "game-abc",
                "commence_time": "2021-10-17T17:00:00Z",  # Sun, 2021 Week 6
                "home_team": "Buffalo Bills",
                "away_team": "Kansas City Chiefs",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 54.5},
                                    {"name": "Under", "point": 54.5},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 55.5},
                                    {"name": "Under", "point": 55.5},
                                ],
                            }
                        ],
                    },
                ],
            }
        ],
    }


def _mock_client(mock_mode: bool = False) -> OddsAPIClient:
    """An OddsAPIClient with HTTP/auth stubbed out (no live call)."""
    client = OddsAPIClient.__new__(OddsAPIClient)
    client.mock_mode = mock_mode
    client.api_key = None if mock_mode else "test-key"
    return client


def test_backfill_stamps_envelope_timestamp_and_is_idempotent(tmp_path):
    """Stored snapshot_ts == envelope timestamp (not requested T); re-run no-op."""
    client = _mock_client(mock_mode=False)

    requested_ts: list[str] = []

    def fake_hist(date_iso, markets, regions="us"):
        requested_ts.append(date_iso)
        return _fake_envelope()

    client.get_historical_nfl_odds = fake_hist

    written = backfill_timeline([2021], client, weeks=[6], base_path=tmp_path)
    assert written > 0

    silver = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")

    # The four cadence timestamps were requested, but NONE equals the envelope
    # timestamp -- proving provenance is taken from the envelope (review 29-03 HIGH).
    assert len(requested_ts) == 4
    assert ENVELOPE_TS not in requested_ts

    # Exactly one consensus row at the (game_id, snapshot_ts) grain, stamped on
    # the envelope timestamp.
    assert len(silver) == 1
    row = silver.iloc[0]
    assert row["game_id"] == "2021_W06_KC@BUF"
    assert row["total"] == pytest.approx(55.0)  # median(54.5, 55.5)
    stored_ts = pd.Timestamp(row["snapshot_ts"])
    assert stored_ts == pd.Timestamp("2021-10-15T21:57:00+00:00")

    # Double-run idempotency on the composite key.
    backfill_timeline([2021], client, weeks=[6], base_path=tmp_path)
    silver_again = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
    assert len(silver_again) == 1


def test_freeze_fence_resolves_to_et_evening_instant():
    """The Fri-18:00-ET freeze is an ET-evening UTC instant, never UTC 18:00."""
    cadence = dict(weekly_snapshot_timestamps(2021, 6))
    freeze_utc = cadence["freeze"]

    assert freeze_utc.tzinfo is not None
    # Friday 18:00 ET in October is EDT (UTC-4) -> 22:00 UTC; never 18:00 UTC.
    assert freeze_utc.hour in (22, 23)
    assert freeze_utc.hour != 18

    # Round-tripping back to ET yields exactly Friday 18:00.
    freeze_et = freeze_utc.astimezone(ET)
    assert freeze_et.hour == 18
    assert freeze_et.minute == 0
    assert freeze_et.weekday() == 4  # Friday

    # A January (winter / EST) week resolves to 23:00 UTC.
    winter = dict(weekly_snapshot_timestamps(2021, 18))
    assert winter["freeze"].astimezone(ET).hour == 18


def test_backfill_hard_fails_on_mock_mode():
    """Backfill refuses a mock-mode client before any call (OUM-06)."""
    client = _mock_client(mock_mode=True)
    sentinel = MagicMock()
    client.get_historical_nfl_odds = sentinel

    with pytest.raises(MockModeBackfillError):
        backfill_timeline([2021], client, weeks=[6])

    sentinel.assert_not_called()


def test_raw_envelope_normalizer_builds_consensus_rows():
    """The dedicated normalizer turns a raw envelope into consensus rows."""
    rows = normalize_envelope_to_timeline_rows(_fake_envelope(), ["totals"])

    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == "2021_W06_KC@BUF"
    assert row["total"] == pytest.approx(55.0)
    assert row["spread"] is None  # spreads not requested
    assert row["sportsbook"] == "consensus_median"
    assert row["region"] == "us"
    # snapshot_ts taken from the envelope timestamp (review 29-03 HIGH).
    assert row["snapshot_ts"] == datetime(2021, 10, 15, 21, 57, tzinfo=UTC)


def test_debug_params_log_redacts_api_key():
    """The _make_request DEBUG params log never contains the literal apiKey."""
    secret = "SUPER_SECRET_KEY_98765"
    client = OddsAPIClient.__new__(OddsAPIClient)
    client.mock_mode = False
    client.api_key = secret
    client.base_url = "https://api.example.com/v4"

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"data": []}
    fake_response.status_code = 200
    client.client = MagicMock()
    client.client.get.return_value = fake_response

    with patch("scripts.ingest_odds.logger") as mock_logger:
        client._make_request("some/endpoint", {"regions": "us"})

    debug_calls = list(mock_logger.debug.call_args_list)
    assert debug_calls, "expected a DEBUG params log call"
    logged = str(debug_calls)
    assert secret not in logged
    assert "***REDACTED***" in logged
