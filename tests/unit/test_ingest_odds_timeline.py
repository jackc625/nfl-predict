"""Mock/stub-mode tests for scripts/ingest_odds_timeline.py (SIG-04).

These tests NEVER make a live (paid) historical call -- the real paid pull is
Plan 29-05 under a paid key. They prove:

1. The backfill loop stamps ``snapshot_ts`` from the ENVELOPE timestamp, NOT the
   requested date T, and the composite-key write is idempotent on a double run
   (review 29-03 HIGH).
2. A per-game LOCK instant resolves to an ET-evening instant (22:00/23:00 UTC),
   never a UTC-localized 18:00 (WR-02) -- retargeted by Plan 33.2-02 from the single
   week-level Friday freeze the cadence used to carry to each game's own day-before
   lock. The collection CADENCE stays week-level (three samples); ADMISSIBILITY is
   per game, so the backfill requests the cadence plus the week's distinct locks.
3. The backfill path HARD-FAILS on a mock-mode client (OUM-06).
4. The dedicated raw-envelope normalizer turns a raw API game envelope into
   consensus ``OddsTimelineSchema`` rows (review 29-03 MED).
5. ``_make_request``'s DEBUG params log redacts ``apiKey`` (review 29-03 MED).
6. SPEND SAFETY (Plan 29-05): a re-run of an already-pulled slice issues ZERO
   paid calls, and the skip guard matches the stored ENVELOPE timestamp rather
   than the requested T (an equality check would be a silent no-op guard).
7. SPEND SAFETY (Plan 29-05): ``--weeks`` is exposed on the CLI so a paid pull
   can be bounded to a few timestamps.
8. SPEND CEILING (WR-03): the credit-header guard and the paid-call ceiling are
   WIRED INTO the loop and abort it. Before the fix the header reader had zero
   call sites in the repo -- production or test -- while the module's prose
   credited it as "the cost guard", and nothing bounded total spend.
9. SPEND SAFETY (WR-04): a paid call that yields ZERO usable rows still records
   coverage from the envelope timestamp, so a resumed backfill does not re-buy
   it forever.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.ingest_odds import OddsAPIClient
from scripts.ingest_odds_timeline import (
    _GAMES_ID_CACHE,
    MockModeBackfillError,
    SpendGuardError,
    _build_parser,
    _load_stored_snapshot_timestamps,
    _snapshot_already_stored,
    backfill_timeline,
    game_lock_instants,
    normalize_envelope_to_timeline_rows,
    weekly_cadence_timestamps,
)
from utils import DataIngestionError
from utils.game_lock import MissingKickoffError, game_lock

ET = ZoneInfo("America/New_York")

# The envelope's actual snapshot timestamp -- deliberately distinct from any of
# the requested weekly cadence timestamps T (review 29-03 HIGH).
ENVELOPE_TS = "2021-10-15T21:57:00Z"

# The one game of the 2021 week-6 fixture: Sunday 2021-10-17 1 PM ET. Its lock is
# Saturday 2021-10-16 18:00 ET = 22:00 UTC, so the week's requests are the three
# cadence samples (Tue/Wed/Thu noon ET) plus that one lock instant: FOUR.
WEEK6_GAMES = pd.DataFrame(
    {
        "game_id": ["2021_W06_KC@BUF"],
        "season": [2021],
        "week": [6],
        "home_team": ["BUF"],
        "away_team": ["KC"],
        "kickoff_et": [pd.Timestamp("2021-10-17T17:00:00Z")],
    }
)
WEEK6_LOCK_UTC = datetime(2021, 10, 16, 22, 0, tzinfo=UTC)


def _week_games(season: int, week: int, *kickoffs_utc: str) -> pd.DataFrame:
    """A schedule slice with one game per kickoff (ids are placeholders)."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_W{week:02d}_G{i}" for i in range(len(kickoffs_utc))],
            "season": [season] * len(kickoffs_utc),
            "week": [week] * len(kickoffs_utc),
            "kickoff_et": [pd.Timestamp(k) for k in kickoffs_utc],
        }
    )


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


def _fake_headers(remaining: int = 20000, last: int = 10) -> dict[str, str]:
    """Synthetic Odds API credit headers (WR-03).

    The cost guard is exercised entirely against these -- no live call is needed
    to prove the abort arms fire.
    """
    return {"x-requests-remaining": str(remaining), "x-requests-last": str(last)}


def test_backfill_stamps_envelope_timestamp_and_is_idempotent(tmp_path):
    """Stored snapshot_ts == envelope timestamp (not requested T); re-run no-op."""
    client = _mock_client(mock_mode=False)

    requested_ts: list[str] = []

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        requested_ts.append(date_iso)
        envelope = _fake_envelope()
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    written = backfill_timeline(
        [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
    )
    assert written > 0

    silver = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")

    # The week's union is three cadence samples plus the Sunday game's Saturday
    # 18:00 ET lock, and all four are requested: the fake's CONSTANT envelope
    # timestamp (Fri 21:57Z) lies outside every request's lookback window, the
    # lock's included (it is 24h earlier). NONE of the requested T's equals the
    # envelope timestamp -- proving provenance is taken from the envelope
    # (review 29-03 HIGH).
    assert len(requested_ts) == 4
    assert ENVELOPE_TS not in requested_ts
    assert WEEK6_LOCK_UTC.strftime("%Y-%m-%dT%H:%M:%SZ") in requested_ts

    # Exactly one consensus row at the (game_id, snapshot_ts) grain, stamped on
    # the envelope timestamp.
    assert len(silver) == 1
    row = silver.iloc[0]
    assert row["game_id"] == "2021_W06_KC@BUF"
    assert row["total"] == pytest.approx(55.0)  # median(54.5, 55.5)
    stored_ts = pd.Timestamp(row["snapshot_ts"])
    assert stored_ts == pd.Timestamp("2021-10-15T21:57:00+00:00")

    # Double-run idempotency on the composite key.
    backfill_timeline([2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES)
    silver_again = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
    assert len(silver_again) == 1


def test_a_per_game_lock_resolves_to_an_et_evening_instant():
    """A game's lock is an ET-evening UTC instant, never UTC 18:00 (WR-02).

    October (EDT, UTC-4) resolves to 22:00 UTC and January (EST, UTC-5) to 23:00 UTC;
    both round-trip to exactly 18:00 ET on the ET day before kickoff.
    """
    october = game_lock_instants(WEEK6_GAMES)
    assert len(october) == 1
    label, game_ids, lock_utc = october[0]
    assert label == "lock"
    assert game_ids == "2021_W06_KC@BUF"
    assert lock_utc.tzinfo is not None
    assert lock_utc.hour == 22
    assert lock_utc.hour != 18
    lock_et = lock_utc.astimezone(ET)
    assert (lock_et.hour, lock_et.minute, lock_et.weekday()) == (18, 0, 5)  # Saturday

    january = game_lock_instants(_week_games(2021, 18, "2022-01-09T18:00:00Z"))
    (_label, _ids, winter_utc) = january[0]
    assert winter_utc.hour == 23
    assert winter_utc.astimezone(ET).hour == 18


def test_the_cadence_is_three_week_level_samples_with_no_freeze():
    cadence = weekly_cadence_timestamps(2021, 6)

    assert sorted(label for label, _ in cadence) == ["intraweek", "late", "open"]
    for _label, instant in cadence:
        assert instant.tzinfo is not None
        assert instant.astimezone(ET).hour == 12


def test_lock_instants_are_the_one_rule_and_collapse_by_instant():
    """A week's Sunday games share ONE lock, so they cost ONE request, not one each."""
    games = _week_games(
        2021,
        6,
        "2021-10-15T00:20:00Z",  # Thursday night 8:20 PM ET -> Wed lock
        "2021-10-17T17:00:00Z",  # Sunday 1 PM ET -> Sat lock
        "2021-10-17T20:25:00Z",  # Sunday 4:25 PM ET -> the SAME Sat lock
        "2021-10-19T00:15:00Z",  # Monday night 8:15 PM ET -> Sun lock
    )

    instants = game_lock_instants(games)

    assert len(instants) == 3, "the two Sunday games must collapse into one lock"
    kickoff_by_id = dict(zip(games["game_id"], games["kickoff_et"], strict=True))
    for _label, game_ids, lock_utc in instants:
        for game_id in game_ids.split(","):
            assert lock_utc == game_lock(kickoff_by_id[game_id])
    assert [lock for _, _, lock in instants] == sorted(lock for _, _, lock in instants)


def test_lock_instants_refuse_a_game_with_no_kickoff():
    games = _week_games(2021, 6, "2021-10-17T17:00:00Z", "2021-10-17T20:25:00Z")
    games.loc[1, "kickoff_et"] = pd.NaT

    with pytest.raises(MissingKickoffError, match="2021_W06_G1"):
        game_lock_instants(games)


def test_backfill_hard_fails_on_mock_mode():
    """Backfill refuses a mock-mode client before any call (OUM-06)."""
    client = _mock_client(mock_mode=True)
    sentinel = MagicMock()
    client.get_historical_nfl_odds = sentinel

    with pytest.raises(MockModeBackfillError):
        backfill_timeline([2021], client, weeks=[6], games=WEEK6_GAMES)

    sentinel.assert_not_called()


def test_raw_envelope_normalizer_builds_consensus_rows():
    """The dedicated normalizer turns a raw envelope into consensus rows."""
    rows = normalize_envelope_to_timeline_rows(
        _fake_envelope(), ["totals"], schedule=WEEK6_GAMES
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["game_id"] == "2021_W06_KC@BUF"
    assert row["total"] == pytest.approx(55.0)
    assert row["spread"] is None  # spreads not requested
    assert row["sportsbook"] == "consensus_median"
    assert row["region"] == "us"
    # snapshot_ts taken from the envelope timestamp (review 29-03 HIGH).
    assert row["snapshot_ts"] == datetime(2021, 10, 15, 21, 57, tzinfo=UTC)


def _fake_envelope_at(requested_iso: str) -> dict:
    """An envelope whose timestamp is 5 minutes EARLIER than the requested T.

    Mirrors the observed live behaviour (requested ``2021-10-15T22:00:00Z`` ->
    envelope ``2021-10-15T21:55:00Z``): the API returns the closest archived
    snapshot at/earlier than T, so a stored ``snapshot_ts`` NEVER equals T.
    """
    requested = datetime.fromisoformat(requested_iso.replace("Z", "+00:00"))
    envelope = _fake_envelope()
    envelope["timestamp"] = (requested - timedelta(minutes=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return envelope


def test_backfill_skips_already_stored_snapshots_on_rerun(tmp_path):
    """A re-run of an already-pulled slice issues ZERO paid calls (spend safety).

    ``upsert_silver_composite`` makes the WRITE idempotent but not the paid
    CALL; without the skip guard, resuming an interrupted backfill would re-buy
    every timestamp already on disk.
    """
    client = _mock_client(mock_mode=False)

    first_calls: list[str] = []

    def fake_hist_first(date_iso, markets, regions="us", return_headers=False):
        first_calls.append(date_iso)
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist_first

    backfill_timeline([2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES)

    silver_path = tmp_path / "silver" / "odds_timeline.parquet"
    silver = pd.read_parquet(silver_path)
    assert len(first_calls) == 4  # three cadence samples + the game's lock
    assert len(silver) == 4
    assert silver["snapshot_ts"].nunique() == 4

    # The guard cannot be an equality check: no stored timestamp equals any
    # requested T, so an equality-based guard would silently never fire.
    stored_iso = {
        pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ") for ts in silver["snapshot_ts"]
    }
    assert stored_iso.isdisjoint(set(first_calls))

    second_calls: list[str] = []

    def fake_hist_second(date_iso, markets, regions="us", return_headers=False):
        second_calls.append(date_iso)
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist_second

    written = backfill_timeline(
        [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
    )

    # ZERO additional API calls -> zero additional credits burned.
    assert second_calls == []
    assert written == 0
    assert len(pd.read_parquet(silver_path)) == 4


def test_spend_guard_is_markets_aware_on_a_synthetic_archive(tmp_path):
    """WR-03: a timestamp counts as covered only for the markets already stored.

    The guard previously read the ``snapshot_ts`` column alone, so a
    ``--markets totals`` run followed by ``--markets totals spreads`` would find
    every timestamp present, skip all of them, make ZERO calls, write zero rows and
    leave ``spread`` null forever -- precisely the resumed-backfill case the guard
    exists to serve.

    SYNTHETIC ONLY: this writes a small parquet by hand and never constructs an
    ``OddsAPIClient`` or exercises a real backfill. The live archive cost 7,210 real
    credits and is never touched by a test.
    """
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True)
    stored_ts = pd.Timestamp("2021-10-15T21:55:00Z")
    pd.DataFrame(
        {
            "game_id": ["2021_W06_A@B"],
            "snapshot_ts": [stored_ts],
            "total": [44.0],
            "spread": [None],  # totals-only run: the spread column is null
        }
    ).to_parquet(silver_dir / "odds_timeline.parquet", index=False)

    totals_only = _load_stored_snapshot_timestamps(tmp_path, markets=["totals"])
    assert stored_ts in totals_only, "a stored total must count as totals coverage"

    with_spreads = _load_stored_snapshot_timestamps(
        tmp_path, markets=["totals", "spreads"]
    )
    assert with_spreads == set(), (
        "a timestamp with a null spread must NOT count as covered when spreads are "
        "requested, or the widened run silently skips and never fills the column"
    )

    requested = datetime(2021, 10, 15, 22, 0, tzinfo=UTC)
    assert _snapshot_already_stored(requested, totals_only) is True
    assert _snapshot_already_stored(requested, with_spreads) is False


def test_spend_guard_counts_a_timestamp_covered_when_all_markets_present(tmp_path):
    """The other direction: both markets stored means the skip correctly fires."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir(parents=True)
    stored_ts = pd.Timestamp("2021-10-15T21:55:00Z")
    pd.DataFrame(
        {
            "game_id": ["2021_W06_A@B"],
            "snapshot_ts": [stored_ts],
            "total": [44.0],
            "spread": [-3.5],
        }
    ).to_parquet(silver_dir / "odds_timeline.parquet", index=False)

    assert _load_stored_snapshot_timestamps(
        tmp_path, markets=["totals", "spreads"]
    ) == {stored_ts}


def test_spend_guard_returns_empty_set_when_the_archive_does_not_exist(tmp_path):
    """First run: nothing stored, nothing skipped."""
    assert _load_stored_snapshot_timestamps(tmp_path, markets=["totals"]) == set()


def test_snapshot_guard_window_matches_earlier_envelope_not_prior_cadence():
    """The coverage window catches a slightly-earlier envelope, never a neighbour."""
    requested = datetime(2021, 10, 15, 22, 0, tzinfo=UTC)

    # Fires for the realistic 5-minute-earlier envelope and for an exact match.
    assert _snapshot_already_stored(requested, {pd.Timestamp("2021-10-15T21:55:00Z")})
    assert _snapshot_already_stored(requested, {pd.Timestamp("2021-10-15T22:00:00Z")})

    # Does not fire outside the window, for a later snapshot, or on empty state.
    assert not _snapshot_already_stored(
        requested, {pd.Timestamp("2021-10-15T09:00:00Z")}
    )
    assert not _snapshot_already_stored(
        requested, {pd.Timestamp("2021-10-15T22:05:00Z")}
    )
    assert not _snapshot_already_stored(requested, set())

    # No FALSE skip between cadence samples: consecutive ones are 24h apart.
    cadence = dict(weekly_cadence_timestamps(2021, 6))
    assert not _snapshot_already_stored(
        cadence["late"], {pd.Timestamp(cadence["intraweek"])}
    )

    # No FALSE skip between a cadence sample and a LOCK. The closest pairs in the
    # union are only SIX hours apart: a Thursday game locks Wednesday 18:00 ET,
    # six hours after the Wednesday-noon sample, and a Friday game locks Thursday
    # 18:00 ET, six hours after the Thursday-noon sample. A lookback of six hours
    # or more would treat the noon snapshot as covering the lock and never buy it.
    thursday_game_lock = game_lock_instants(
        _week_games(2021, 6, "2021-10-15T00:20:00Z")
    )[0][2]
    friday_game_lock = game_lock_instants(_week_games(2021, 6, "2021-10-16T00:20:00Z"))[
        0
    ][2]
    assert thursday_game_lock - cadence["intraweek"] == timedelta(hours=6)
    assert friday_game_lock - cadence["late"] == timedelta(hours=6)
    assert not _snapshot_already_stored(
        thursday_game_lock,
        {pd.Timestamp(cadence["intraweek"]) - pd.Timedelta(minutes=5)},
    )
    assert not _snapshot_already_stored(
        friday_game_lock, {pd.Timestamp(cadence["late"]) - pd.Timedelta(minutes=5)}
    )


def test_cli_exposes_weeks_flag_for_bounded_paid_pulls():
    """``--weeks`` bounds a paid pull (smoke check / targeted recovery)."""
    args = _build_parser().parse_args(
        ["--backfill", "2020", "2020", "--markets", "totals", "spreads", "--weeks", "1"]
    )
    assert args.backfill == [2020, 2020]
    assert args.weeks == [1]
    assert args.markets == ["totals", "spreads"]

    # Omitting --weeks keeps the full 1-18 season default.
    assert _build_parser().parse_args(["--backfill", "2020", "2024"]).weeks is None


def test_normalizer_maps_the_2020_washington_football_team_name():
    """The 2020-2021 franchise name resolves to canonical WAS (no hard-fail).

    The historical archive sends "Washington Football Team" for 2020-2021; an
    unmapped name aborts the paid backfill mid-loop.
    """
    envelope = {
        "timestamp": "2020-09-13T16:55:00Z",
        "data": [
            {
                "id": "game-wft",
                "commence_time": "2020-09-13T17:00:00Z",
                "home_team": "Washington Football Team",
                "away_team": "Philadelphia Eagles",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 44.5},
                                    {"name": "Under", "point": 44.5},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    schedule = pd.DataFrame(
        {
            "game_id": ["2020_W01_PHI@WAS"],
            "season": [2020],
            "week": [1],
            "home_team": ["WAS"],
            "away_team": ["PHI"],
            "kickoff_et": [pd.Timestamp("2020-09-13T17:00:00Z")],
        }
    )

    rows = normalize_envelope_to_timeline_rows(envelope, ["totals"], schedule=schedule)

    assert len(rows) == 1
    assert rows[0]["game_id"] == "2020_W01_PHI@WAS"


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


# ---------------------------------------------------------------------------
# WR-03: the credit-header cost guard and the paid-call ceiling
#
# Repo-wide grep found `return_headers=True` and the header-reading request
# method had ZERO call sites outside their own definitions -- no production
# caller, no test -- while the module docstring credited them as "the cost
# guard". `backfill_timeline` called the JSON-only path, never read a credit
# header, and had no cap on `calls_made`. A wrong `--backfill 2015 2024` issued
# 18 weeks x 4 cadence points x N seasons paid calls with nothing between the
# loop and the account balance.
# ---------------------------------------------------------------------------


def _counting_client(headers_for_call, tmp_path=None):
    """A stub client that returns a fresh envelope + caller-chosen headers."""
    client = _mock_client(mock_mode=False)
    calls: list[str] = []

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        calls.append(date_iso)
        envelope = _fake_envelope_at(date_iso)
        if not return_headers:
            return envelope
        return envelope, headers_for_call(len(calls))

    client.get_historical_nfl_odds = fake_hist
    return client, calls


def test_backfill_reads_the_credit_headers_on_every_paid_call(tmp_path):
    """The guard is WIRED IN: the loop requests headers, not the JSON-only path."""
    seen_return_headers: list[bool] = []
    client = _mock_client(mock_mode=False)

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        seen_return_headers.append(return_headers)
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    backfill_timeline([2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES)

    assert seen_return_headers, "no paid call was made"
    assert all(seen_return_headers), (
        "the backfill called the JSON-only path, so the credit headers are "
        "never read and the cost guard is dead code (WR-03)"
    )


def test_backfill_aborts_when_credits_fall_below_the_floor(tmp_path):
    """A credit floor stops the loop rather than draining the account."""
    client, calls = _counting_client(
        lambda n: _fake_headers(remaining=5 if n >= 2 else 20000)
    )

    with pytest.raises(SpendGuardError, match="credits"):
        backfill_timeline(
            [2021],
            client,
            weeks=[6],
            base_path=tmp_path,
            min_credits_remaining=100,
            games=WEEK6_GAMES,
        )

    # Stopped ON the offending call, not after burning the rest of the cadence.
    assert len(calls) == 2


def test_backfill_aborts_at_the_paid_call_ceiling(tmp_path):
    """A hard ceiling bounds a typo'd multi-season backfill."""
    client, calls = _counting_client(lambda _n: _fake_headers())

    with pytest.raises(SpendGuardError, match="ceiling"):
        backfill_timeline(
            [2021],
            client,
            weeks=[6, 7, 8],
            base_path=tmp_path,
            max_paid_calls=3,
            games=WEEK6_GAMES,
        )

    assert len(calls) == 3


def test_rows_bought_before_an_abort_are_kept(tmp_path):
    """The guard fires AFTER the write, so a paid call is never discarded.

    Aborting before persisting would mean the operator paid for a snapshot and
    threw it away -- and the re-run would then buy it a second time.
    """
    client, _calls = _counting_client(lambda _n: _fake_headers())

    with pytest.raises(SpendGuardError):
        backfill_timeline(
            [2021],
            client,
            weeks=[6],
            base_path=tmp_path,
            max_paid_calls=2,
            games=WEEK6_GAMES,
        )

    silver = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
    assert len(silver) == 2


def test_a_missing_credit_header_does_not_abort_a_legitimate_backfill(tmp_path):
    """An absent header reads as UNKNOWN, not as zero credits.

    The call ceiling still bounds the run, so treating a missing header as an
    abort would only break legitimate backfills against a proxy that strips it.
    """
    client, calls = _counting_client(lambda _n: {})

    backfill_timeline([2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES)

    assert len(calls) == 4


# ---------------------------------------------------------------------------
# WR-04: a paid call yielding zero usable rows must not be re-bought
# ---------------------------------------------------------------------------


def _empty_board_envelope(requested_iso: str) -> dict:
    """An envelope with a real timestamp but NO games (nothing to normalize)."""
    envelope = _fake_envelope_at(requested_iso)
    envelope["data"] = []
    return envelope


def test_an_empty_envelope_still_records_coverage_within_the_run(tmp_path):
    """Coverage comes from the ENVELOPE timestamp, not from the emitted rows.

    Pre-fix, an envelope with no usable board added nothing to
    ``stored_snapshots`` and wrote nothing, so a later requested timestamp that
    the SAME archived snapshot already answers was bought all over again.

    The fixture returns a CONSTANT envelope timestamp three minutes before the
    Sunday game's Saturday 18:00 ET lock (21:57Z Saturday), which the lookback
    window means also covers that lock request. Four requested instants (three
    cadence samples and the lock), three purchases: the lock is skipped because
    an earlier call's envelope already answered it. Pre-fix that skip could not
    happen for an empty board and all four were bought.

    SCOPE, stated rather than implied: this record is IN-MEMORY, so it bounds
    re-spend within one invocation only. It is deliberately not durable -- a
    zero-row marker or a sidecar "requested timestamps" file would be needed for
    that, and adding an artifact to the data lake is a design decision, not a
    review fix. The WARNING log below is what makes the event visible meanwhile.
    """
    client = _mock_client(mock_mode=False)
    requested: list[str] = []

    lock_adjacent_ts = "2021-10-16T21:57:00Z"

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        requested.append(date_iso)
        envelope = _fake_envelope(lock_adjacent_ts)  # constant, no usable board
        envelope["data"] = []
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    written = backfill_timeline(
        [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
    )

    assert written == 0
    assert len(requested) == 3, (
        f"expected the covered lock request to be skipped, bought "
        f"{len(requested)} -- coverage is still being recorded from the emitted "
        f"rows instead of the envelope timestamp (WR-04)"
    )
    assert WEEK6_LOCK_UTC.strftime("%Y-%m-%dT%H:%M:%SZ") not in requested
    assert lock_adjacent_ts not in requested


def test_an_empty_envelope_is_logged_at_warning(tmp_path):
    """A paid call that bought nothing is exactly what an operator must see."""
    client = _mock_client(mock_mode=False)

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        envelope = _empty_board_envelope(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    with patch("scripts.ingest_odds_timeline.logger") as mock_logger:
        backfill_timeline(
            [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
        )

    warnings_logged = str(mock_logger.warning.call_args_list)
    assert "NO usable rows" in warnings_logged


# ---------------------------------------------------------------------------
# WR-11: one request implementation, two return shapes
# ---------------------------------------------------------------------------


def test_make_request_returns_headers_on_demand():
    """``with_headers=True`` returns ``(json, headers)`` from the SAME method.

    The header-returning path used to be a ~45-line near-verbatim duplicate with
    no test at all, so any change to auth, retry policy or error mapping had to
    be made twice.
    """
    client = OddsAPIClient.__new__(OddsAPIClient)
    client.mock_mode = False
    client.api_key = "k"
    client.base_url = "https://api.example.com/v4"

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"timestamp": ENVELOPE_TS, "data": []}
    fake_response.status_code = 200
    fake_response.headers = {"x-requests-remaining": "19990"}
    client.client = MagicMock()
    client.client.get.return_value = fake_response

    plain = client._make_request("some/endpoint", {"regions": "us"})
    body, headers = client._make_request(
        "some/endpoint", {"regions": "us"}, with_headers=True
    )

    assert plain == {"timestamp": ENVELOPE_TS, "data": []}
    assert body == plain
    assert headers["x-requests-remaining"] == "19990"


def test_make_request_does_not_retry_a_programming_error():
    """A bare ``except Exception`` turned bugs into three PAID retries.

    Anything outside the malformed-body family now propagates un-retried.
    """
    client = OddsAPIClient.__new__(OddsAPIClient)
    client.mock_mode = False
    client.api_key = "k"
    client.base_url = "https://api.example.com/v4"

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.side_effect = MemoryError("not a transport failure")
    client.client = MagicMock()
    client.client.get.return_value = fake_response

    with pytest.raises(MemoryError):
        client._make_request("some/endpoint")

    assert client.client.get.call_count == 1, (
        "a non-recoverable error was retried; each retry is a PAID HTTP call"
    )


# ---------------------------------------------------------------------------
# WR-05: a listing that cannot be placed is refused, never clamped or guessed
#
# The week used to be DERIVED arithmetically from commence_time and was never
# RECONCILED against games silver -- the exact mechanism that orphaned the whole
# 2020 archive (D29-06-01) and cost a re-key of 1,780 paid rows. A max(1, min(week,
# 22)) clamp once turned an out-of-range date into a VALID-LOOKING key: a listing
# before the season opener became {season}_W01_AWAY@HOME, which can collide with the
# real Week-1 meeting of the same two teams. Because upsert_silver_composite dedupes
# with keep="last", a same-timestamp collision silently overwrites a real paid row.
#
# Plan 33.2-24 step 24b retired the derivation itself: every event is now keyed to its
# scheduled game (tests/unit/test_odds_timeline_schedule_keying.py). These four tests
# keep WR-05's intent -- refuse, never clamp -- against the schedule match. Was: three
# of them called the deleted _derive_season_week directly.
# ---------------------------------------------------------------------------


def _one_game_board(commence_time: str) -> dict:
    envelope = _fake_envelope()
    envelope["data"][0]["commence_time"] = commence_time
    return envelope


def test_an_out_of_season_listing_is_refused_not_clamped():
    """A pre-opener listing must not become the real Week-1 (or any) game_id."""
    rows = normalize_envelope_to_timeline_rows(
        _one_game_board("2021-07-04T17:00:00Z"), ["totals"], schedule=WEEK6_GAMES
    )
    assert rows == []


def test_a_far_future_listing_is_refused_not_clamped():
    """Nor may a listing far past the schedule be clamped onto its last meeting."""
    rows = normalize_envelope_to_timeline_rows(
        _one_game_board("2022-07-20T17:00:00Z"), ["totals"], schedule=WEEK6_GAMES
    )
    assert rows == []


def test_a_normal_in_season_listing_still_keys():
    """Positive control: the refusal must not break the ordinary path."""
    rows = normalize_envelope_to_timeline_rows(
        _one_game_board("2021-10-17T17:00:00Z"), ["totals"], schedule=WEEK6_GAMES
    )
    assert [row["game_id"] for row in rows] == ["2021_W06_KC@BUF"]


def test_an_out_of_season_game_is_skipped_without_discarding_the_board():
    """One bad listing must not throw away the rest of a PAID snapshot."""
    envelope = _fake_envelope()
    bad_game = dict(envelope["data"][0])
    bad_game["commence_time"] = "2021-07-04T17:00:00Z"
    bad_game["home_team"] = "Chicago Bears"
    bad_game["away_team"] = "Green Bay Packers"
    envelope["data"] = [bad_game, *envelope["data"]]

    rows = normalize_envelope_to_timeline_rows(
        envelope, ["totals"], schedule=WEEK6_GAMES
    )

    assert len(rows) == 1
    assert rows[0]["game_id"] == "2021_W06_KC@BUF"


def test_orphaned_game_ids_are_reported_at_warning(tmp_path):
    """A derived key that does not join games silver gets ONE warning line.

    That single line would have surfaced D29-06-01 on the day it happened.
    """
    _GAMES_ID_CACHE.clear()
    (tmp_path / "silver").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": ["2021_W06_SOMETHING_ELSE"]}).to_parquet(
        tmp_path / "silver" / "games.parquet"
    )

    client = _mock_client(mock_mode=False)

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    with patch("scripts.ingest_odds_timeline.logger") as mock_logger:
        backfill_timeline(
            [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
        )

    logged = str(mock_logger.warning.call_args_list)
    _GAMES_ID_CACHE.clear()

    assert "do NOT join games silver" in logged
    assert "2021_W06_KC@BUF" in logged


def test_no_orphan_warning_when_the_key_joins(tmp_path):
    """Positive control: a joining key must be silent, or the warning is noise."""
    _GAMES_ID_CACHE.clear()
    (tmp_path / "silver").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": ["2021_W06_KC@BUF"]}).to_parquet(
        tmp_path / "silver" / "games.parquet"
    )

    client = _mock_client(mock_mode=False)

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    with patch("scripts.ingest_odds_timeline.logger") as mock_logger:
        backfill_timeline(
            [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
        )

    logged = str(mock_logger.warning.call_args_list)
    _GAMES_ID_CACHE.clear()

    assert "do NOT join games silver" not in logged


def test_a_missing_games_table_does_not_block_the_write(tmp_path):
    """The check WARNS; it must never turn a missing games table into data loss.

    The paid call has already been made by the time this runs.
    """
    _GAMES_ID_CACHE.clear()
    client = _mock_client(mock_mode=False)

    def fake_hist(date_iso, markets, regions="us", return_headers=False):
        envelope = _fake_envelope_at(date_iso)
        return (envelope, _fake_headers()) if return_headers else envelope

    client.get_historical_nfl_odds = fake_hist

    written = backfill_timeline(
        [2021], client, weeks=[6], base_path=tmp_path, games=WEEK6_GAMES
    )
    _GAMES_ID_CACHE.clear()

    assert written == 4


def test_the_backfill_refuses_to_run_without_a_schedule(tmp_path):
    """No schedule means no per-game lock, and a guessed one is the defect."""
    client = _mock_client(mock_mode=False)
    sentinel = MagicMock()
    client.get_historical_nfl_odds = sentinel

    with pytest.raises(DataIngestionError, match="schedule"):
        backfill_timeline([2021], client, weeks=[6], base_path=tmp_path)

    sentinel.assert_not_called()
