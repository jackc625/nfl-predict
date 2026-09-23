"""A captured odds event is keyed to its game by the SCHEDULE, never by a week count.

THE DEFECT (found by Plan 33.2-24, fixed by its step 24b). ``scripts/ingest_odds_timeline``
used to derive a captured event's week by counting whole weeks from the season's opening
THURSDAY to the event's ``commence_time``. Every game played on a day BEFORE the Thursday of
its own week therefore landed in the PREVIOUS week:

* ``2024_W17_KC@PIT`` and ``2024_W17_BAL@HOU`` (Wednesday 2024-12-25) were stored as
  ``2024_W16_KC@PIT`` / ``2024_W16_BAL@HOU`` -- ids that name no game -- so the owned pre-lock
  lines of both were invisible to the converter fit and the blend corpus;
* ``2026_W12_GB@LA`` (Wednesday 2026-11-25) would be stored as ``2026_W11_GB@LA``;
* ``2026_W01_NE@SEA`` (the Wednesday 2026-09-09 opener) derived week 0 and was REFUSED;
* every Super Bowl since 2021 derives week 23 and was REFUSED, and the 2020 Super Bowl derived
  week 22 where the schedule says 21.

Measured 2026-09-23 over silver ``games`` 2020-2026: exactly those ten games disagree with the
week count. The live 2026 capture path shares the same normalizer, so it carried the same
defect forward into this season.

THE RULE NOW. An event is matched to the silver ``games`` row with the same canonical home and
away teams whose kickoff is NEAREST its ``commence_time``, and only when that kickoff is within
``SCHEDULE_MATCH_TOLERANCE``. The tolerance is below HALF the shortest measured gap between two
meetings of the same ordered pairing (6.16 days), so a match can never be ambiguous. An event
with no such row is REFUSED by name -- the WR-05 shape -- and never keyed by any fallback, because
a fallback is exactly what mis-keyed the two Christmas games silently.
"""

from __future__ import annotations

import ast
import inspect
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.ingest_odds_timeline as timeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"

# Real silver ``games`` rows (measured 2026-09-23), in the shape the ingest reads.
_SCHEDULE_ROWS = [
    ("2020_W01_PHI@WAS", 2020, 1, "WAS", "PHI", "2020-09-13T17:00:00Z"),
    ("2021_W06_KC@BUF", 2021, 6, "BUF", "KC", "2021-10-17T17:00:00Z"),
    ("2022_W18_BAL@CIN", 2022, 18, "CIN", "BAL", "2023-01-08T18:00:00Z"),
    ("2022_W19_BAL@CIN", 2022, 19, "CIN", "BAL", "2023-01-16T01:15:00Z"),
    ("2024_W01_DET@LA", 2024, 1, "LA", "DET", "2024-09-09T00:20:00Z"),
    ("2024_W13_CHI@DET", 2024, 13, "DET", "CHI", "2024-11-28T17:30:00Z"),
    ("2024_W16_HOU@KC", 2024, 16, "KC", "HOU", "2024-12-21T18:00:00Z"),
    ("2024_W16_DET@CHI", 2024, 16, "CHI", "DET", "2024-12-22T18:00:00Z"),
    ("2024_W17_KC@PIT", 2024, 17, "PIT", "KC", "2024-12-25T18:00:00Z"),
    ("2024_W17_BAL@HOU", 2024, 17, "HOU", "BAL", "2024-12-25T21:30:00Z"),
    ("2024_W22_KC@PHI", 2024, 22, "PHI", "KC", "2025-02-09T23:30:00Z"),
    ("2026_W01_NE@SEA", 2026, 1, "SEA", "NE", "2026-09-10T00:20:00Z"),
    ("2026_W12_GB@LA", 2026, 12, "LA", "GB", "2026-11-26T01:00:00Z"),
]

SCHEDULE = pd.DataFrame(
    {
        "game_id": [r[0] for r in _SCHEDULE_ROWS],
        "season": [r[1] for r in _SCHEDULE_ROWS],
        "week": [r[2] for r in _SCHEDULE_ROWS],
        "home_team": [r[3] for r in _SCHEDULE_ROWS],
        "away_team": [r[4] for r in _SCHEDULE_ROWS],
        "kickoff_et": [pd.Timestamp(r[5]) for r in _SCHEDULE_ROWS],
    }
)

# The Odds API's full franchise names, as the envelopes carry them.
_API_NAME = {
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "NE": "New England Patriots",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "WAS": "Washington Football Team",
}


def _event(away: str, home: str, commence_time: str, total: float = 44.5) -> dict:
    """One raw API game envelope entry with one book quoting a total and a spread."""
    home_name = _API_NAME[home]
    return {
        "id": f"evt-{away}-{home}-{commence_time}",
        "commence_time": commence_time,
        "home_team": home_name,
        "away_team": _API_NAME[away],
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": total},
                            {"name": "Under", "point": total},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home_name, "point": -2.5},
                            {"name": _API_NAME[away], "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }


def _envelope(*events: dict, timestamp: str = "2024-12-24T16:55:38Z") -> dict:
    return {"timestamp": timestamp, "data": list(events)}


def _keyed_ids(envelope: dict, schedule: pd.DataFrame = SCHEDULE) -> list[str]:
    rows = timeline.normalize_envelope_to_timeline_rows(
        envelope, ["totals", "spreads"], schedule=schedule
    )
    return [row["game_id"] for row in rows]


class TestTheChristmasGamesKeyToTheScheduleWeek:
    """The defect's own two games, keyed from the week-16 cadence board they sat on."""

    def test_kc_at_pit_on_christmas_day_keys_to_week_17(self):
        ids = _keyed_ids(_envelope(_event("KC", "PIT", "2024-12-25T18:00:00Z")))
        assert ids == ["2024_W17_KC@PIT"]

    def test_bal_at_hou_on_christmas_day_keys_to_week_17(self):
        ids = _keyed_ids(_envelope(_event("BAL", "HOU", "2024-12-25T21:30:00Z")))
        assert ids == ["2024_W17_BAL@HOU"]

    def test_a_board_carrying_both_weeks_keys_each_game_to_its_own_week(self):
        """A Tuesday cadence board lists the Saturday, Sunday AND Christmas games at once."""
        ids = _keyed_ids(
            _envelope(
                _event("HOU", "KC", "2024-12-21T18:00:00Z"),
                _event("DET", "CHI", "2024-12-22T18:00:00Z"),
                _event("KC", "PIT", "2024-12-25T18:00:00Z"),
                _event("BAL", "HOU", "2024-12-25T21:30:00Z"),
                timestamp="2024-12-17T16:55:38Z",
            )
        )
        assert ids == [
            "2024_W16_HOU@KC",
            "2024_W16_DET@CHI",
            "2024_W17_KC@PIT",
            "2024_W17_BAL@HOU",
        ]


class TestTheOrdinaryDaysStillKeyCorrectly:
    """Controls: the fix must not move a game the week count already got right."""

    def test_an_ordinary_sunday_game(self):
        ids = _keyed_ids(_envelope(_event("DET", "CHI", "2024-12-22T18:00:00Z")))
        assert ids == ["2024_W16_DET@CHI"]

    def test_a_thanksgiving_thursday_game(self):
        ids = _keyed_ids(_envelope(_event("CHI", "DET", "2024-11-28T17:30:00Z")))
        assert ids == ["2024_W13_CHI@DET"]

    def test_a_late_season_saturday_game(self):
        ids = _keyed_ids(_envelope(_event("HOU", "KC", "2024-12-21T18:00:00Z")))
        assert ids == ["2024_W16_HOU@KC"]

    def test_the_2020_franchise_name_still_resolves(self):
        ids = _keyed_ids(
            _envelope(
                _event("PHI", "WAS", "2020-09-13T17:00:00Z"),
                timestamp="2020-09-13T16:55:00Z",
            )
        )
        assert ids == ["2020_W01_PHI@WAS"]


class TestTheLive2026PathIsFixedToo:
    """The live capture shares the normalizer; 2026 carries two Wednesday games."""

    def test_the_wednesday_opener_keys_to_week_1_instead_of_being_refused(self):
        ids = _keyed_ids(
            _envelope(
                _event("NE", "SEA", "2026-09-10T00:20:00Z"),
                timestamp="2026-09-08T16:00:00Z",
            )
        )
        assert ids == ["2026_W01_NE@SEA"]

    def test_the_thanksgiving_eve_game_keys_to_week_12_not_11(self):
        ids = _keyed_ids(
            _envelope(
                _event("GB", "LA", "2026-11-26T01:00:00Z"),
                timestamp="2026-11-24T16:00:00Z",
            )
        )
        assert ids == ["2026_W12_GB@LA"]

    def test_the_super_bowl_keys_to_its_schedule_week_instead_of_being_refused(self):
        ids = _keyed_ids(
            _envelope(
                _event("KC", "PHI", "2025-02-09T23:30:00Z"),
                timestamp="2025-02-07T16:00:00Z",
            )
        )
        assert ids == ["2024_W22_KC@PHI"]


class TestAMatchIsNeverAmbiguousAndNeverGuessed:
    def test_a_rematch_seven_days_apart_keys_each_listing_to_its_own_game(self):
        """2022: BAL@CIN in week 18 and again in the wild-card round, same home team."""
        ids = _keyed_ids(
            _envelope(
                _event("BAL", "CIN", "2023-01-08T18:00:00Z"),
                _event("BAL", "CIN", "2023-01-16T01:15:00Z"),
                timestamp="2023-01-06T16:00:00Z",
            )
        )
        assert ids == ["2022_W18_BAL@CIN", "2022_W19_BAL@CIN"]

    def test_a_speculative_pairing_with_no_scheduled_game_is_refused(self):
        """A wild-card pairing priced before week 18 settled names no game at all.

        DET@LA met in 2024 week 1, four months earlier. Keying the listing to that game
        would attach playoff prices to a September game -- so it is refused, and the rest
        of the board survives.
        """
        with patch.object(timeline, "logger") as mock_logger:
            ids = _keyed_ids(
                _envelope(
                    _event("DET", "LA", "2025-01-12T01:15:00Z"),
                    _event("KC", "PIT", "2024-12-25T18:00:00Z"),
                    timestamp="2025-01-01T16:55:38Z",
                )
            )
        assert ids == ["2024_W17_KC@PIT"]
        assert "no scheduled game" in str(mock_logger.warning.call_args_list)

    def test_a_listing_just_outside_the_tolerance_is_refused(self):
        """Six days after a real meeting is a DIFFERENT game, never that one."""
        ids = _keyed_ids(
            _envelope(
                _event("HOU", "KC", "2024-12-27T18:00:00Z"),
                timestamp="2024-12-20T16:00:00Z",
            )
        )
        assert ids == []

    def test_a_small_drift_between_listing_and_schedule_is_matched(self):
        """A flexed kickoff moves by hours, not days; the schedule's game is the one."""
        ids = _keyed_ids(
            _envelope(
                _event("KC", "PIT", "2024-12-25T21:00:00Z"),
                timestamp="2024-12-20T16:00:00Z",
            )
        )
        assert ids == ["2024_W17_KC@PIT"]

    def test_a_pre_opener_listing_is_refused_not_clamped(self):
        """WR-05's case, refused by the schedule now rather than by week arithmetic."""
        ids = _keyed_ids(
            _envelope(
                _event("KC", "BUF", "2021-07-04T17:00:00Z"),
                _event("KC", "BUF", "2021-10-17T17:00:00Z"),
                timestamp="2021-07-01T16:00:00Z",
            )
        )
        assert ids == ["2021_W06_KC@BUF"]

    def test_the_tolerance_is_below_half_the_shortest_same_pairing_gap(self):
        assert timedelta(days=6.15) > timeline.SCHEDULE_MATCH_TOLERANCE * 2


class TestNoWeekIsCountedFromADate:
    def test_the_week_count_derivation_is_gone(self):
        assert not hasattr(timeline, "_derive_season_week")

    def test_the_normalizer_requires_a_schedule(self):
        parameter = inspect.signature(
            timeline.normalize_envelope_to_timeline_rows
        ).parameters.get("schedule")
        assert parameter is not None
        assert parameter.default is inspect.Parameter.empty

    def test_the_module_never_counts_weeks_from_a_commence_time(self):
        """No ``(... ).days // 7`` survives anywhere in the module's parsed tree."""
        tree = ast.parse(inspect.getsource(timeline))
        floor_by_seven = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 7
        ]
        assert floor_by_seven == []

    def test_the_scan_catches_a_planted_week_count(self):
        """Control: the scan above is not blind to the construct it forbids."""
        planted = ast.parse("week = (kickoff - season_start).days // 7 + 1")
        hits = [
            node
            for node in ast.walk(planted)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.FloorDiv)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 7
        ]
        assert len(hits) == 1


def _write_games(root: Path) -> None:
    (root / "silver").mkdir(parents=True, exist_ok=True)
    SCHEDULE.to_parquet(root / "silver" / "games.parquet", index=False)


class TestBothEntryPointsKeyThroughTheSchedule:
    def test_the_live_capture_writes_the_christmas_game_under_week_17(self, tmp_path):
        _write_games(tmp_path)
        client = type("StubClient", (), {})()
        client.get_nfl_odds = lambda markets: [
            _event("KC", "PIT", "2024-12-25T18:00:00Z")
        ]

        written = timeline.capture_current_week(
            client, markets=["totals"], season=2024, week=17, base_path=tmp_path
        )

        stored = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
        assert written == 1
        assert stored["game_id"].tolist() == ["2024_W17_KC@PIT"]

    def test_the_backfill_keys_against_the_whole_schedule_not_the_week_slice(
        self, tmp_path
    ):
        """A week-16 cadence board lists the Christmas games; they belong to week 17."""
        client = type("StubClient", (), {"mock_mode": False})()

        def fake_hist(date_iso, markets, regions="us", return_headers=False):
            envelope = _envelope(
                _event("KC", "PIT", "2024-12-25T18:00:00Z"), timestamp=date_iso
            )
            headers = {"x-requests-remaining": "20000"}
            return (envelope, headers) if return_headers else envelope

        client.get_historical_nfl_odds = fake_hist

        timeline.backfill_timeline(
            [2024], client, weeks=[16], base_path=tmp_path, games=SCHEDULE
        )

        stored = pd.read_parquet(tmp_path / "silver" / "odds_timeline.parquet")
        assert set(stored["game_id"]) == {"2024_W17_KC@PIT"}


@pytest.mark.skipif(not SILVER_GAMES.exists(), reason="production silver games absent")
class TestAgainstTheProductionSchedule:
    """Every 2020+ game, listed at its own kickoff, keys to its own schedule id."""

    @pytest.fixture(scope="class")
    def games(self) -> pd.DataFrame:
        frame = pd.read_parquet(
            SILVER_GAMES,
            columns=[
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "kickoff_et",
            ],
        )
        return frame[frame["season"] >= 2020].reset_index(drop=True)

    def test_every_game_keys_to_itself(self, games: pd.DataFrame):
        assert len(games) > 1000, "non-vacuity: the production schedule was read"
        miskeyed = []
        for row in games.itertuples(index=False):
            kickoff = pd.Timestamp(row.kickoff_et).tz_convert("UTC")
            match = timeline.match_event_to_schedule(
                home_team=row.home_team,
                away_team=row.away_team,
                commence=kickoff.to_pydatetime(),
                schedule=games,
            )
            if match != row.game_id:
                miskeyed.append((row.game_id, match))
        assert miskeyed == []

    def test_the_tolerance_is_below_half_the_measured_same_pairing_gap(
        self, games: pd.DataFrame
    ):
        kickoffs = pd.to_datetime(games["kickoff_et"], utc=True)
        ordered = games.assign(kickoff=kickoffs).sort_values("kickoff")
        gaps = ordered.groupby(["home_team", "away_team"])["kickoff"].diff().dropna()
        assert gaps.min() > timeline.SCHEDULE_MATCH_TOLERANCE * 2
