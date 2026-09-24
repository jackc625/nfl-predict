"""A live odds pull must store the SAME market values the stored history does.

THE DEFECTS THIS GUARDS
-----------------------
Found by Plan 33-18 on 2026-09-15, before any paid pull, and measured in memory on the
documented Odds API v4 payload shape:

1. MONEYLINES WERE ALWAYS BLANK. ``_extract_market_odds`` wrote each h2h price under a
   team-named key (``ml_atl``, ``ml_den``), which ``OddsSchema`` drops -- so ``ml_home``
   and ``ml_away`` were None on every live row and WP could never be priced from a pull.
2. THE SPREAD SIGN FOLLOWED THE FAVOURITE, NOT THE HOME TEAM. The favourite's point was
   stored as ``spread`` and its price as ``spread_ju_home`` whichever side was home, so
   the stored spread was always negative. The stored history uses POSITIVE = HOME
   FAVOURED (2,130 of 2,140 rows in ``silver/odds_snapshot.parquet``), so the live sign
   was inverted for every home favourite and the juices swapped for every away favourite.

WHY THIS BECAME URGENT RATHER THAN MERELY WRONG. The Plan 33-18 fix that stopped the
silver odds write partitioning (commit ``91df505``) made these values REACHABLE: before
it, a real pull landed in side directories nothing read; after it, the same pull writes
blank moneylines and wrong-sign spreads straight into the file ``models/train.py``
reads. Fixing the extractor protects the next scheduled Friday run, not only the
acceptance run.

WHAT IS AND IS NOT DECIDED HERE. This makes new live rows match the existing stored
convention. Whether the ATS code expects that convention is DEF-31-01 (WINDOWS row 2)
and is deliberately not decided here.

SIDES ARE DECIDED BY TEAM NAME, NEVER BY LIST POSITION. Every fixture below that could
be satisfied by position is also run with its outcomes reversed.

Sandboxed throughout except ``TestTheConventionIsTheStoredHistorys``, which READS the
production odds file (never writes it) and skips with a pinned message when absent.

THE PER-GAME INTERFACE (Plan 33.2-02 Task 2). The ingest used to take ONE
``snapshot_time`` for a whole multi-game request and use it as the API's game window,
as every row's ``snapshot_ts`` and as the fallback for a missing bookmaker time. It now
takes the slate's ``schedule`` and keeps four values apart: the kickoff SELECTION
window, the OBSERVED capture instant (``created_at``), each row's UPSTREAM bookmaker
time (``last_update``, NULL when absent) and each game's own LOCK (``snapshot_ts``),
derived only after the payload game is matched to its schedule row. Every
market-extraction property below is unchanged; the classes at the end pin the split.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import data.storage as storage_mod
import scripts.ingest_odds as ingest_odds_module
import utils.game_lock as lock_rule
from backtest.ou_divergence import dedupe_odds_by_book_preference
from data.storage import ParquetManager
from scripts.ingest_odds import OddsDataIngester

ET = ZoneInfo("America/New_York")

# The instant the stand-in response is "observed". Tests fix the capture instant by
# patching the clock the ingest reads, never by passing a value: a caller-supplied
# capture time is a manufactured timestamp (RESEARCH P1).
CAPTURED_AT = datetime(2026, 9, 18, 16, 4, 11, tzinfo=UTC)

# The Sunday 1 PM ET kickoff both week-2 fixtures carry, and its day-before lock.
WEEK2_SUNDAY_KICKOFF = pd.Timestamp("2026-09-20 17:00:00", tz="UTC")
WEEK2_SUNDAY_LOCK = datetime(2026, 9, 19, 18, 0, tzinfo=ET)

# A week-3 Thursday night game, 8:15 PM ET = 00:15 UTC on FRIDAY. Its lock is the
# Wednesday before -- the ET date, not the UTC one, decides it.
WEEK3_THURSDAY_KICKOFF = pd.Timestamp("2026-09-25 00:15:00", tz="UTC")
WEEK3_THURSDAY_LOCK = datetime(2026, 9, 23, 18, 0, tzinfo=ET)

PRODUCTION_ODDS_PATH = Path("data/silver/odds_snapshot.parquet")
PRODUCTION_ODDS_ABSENT_SKIP = (
    "data/silver/odds_snapshot.parquet is absent on this checkout; the stored-convention "
    "pin needs the real history to read"
)

# The share of stored rows that must follow POSITIVE = HOME FAVOURED for the convention
# to count as the history's. Measured 2,130 of 2,140 = 0.99533 on 2026-09-15; the ten
# exceptions are recorded in WINDOWS and not repaired.
STORED_CONVENTION_MINIMUM_SHARE = 0.99

# A home favourite: Atlanta -3.5 at home.
HOME_FAV = {
    "home": "Atlanta Falcons",
    "away": "Carolina Panthers",
    "home_ml": -180,
    "away_ml": 150,
    "home_point": -3.5,
    "home_spread_price": -112,
    "away_spread_price": -108,
    "total": 44.5,
    "over_price": -105,
    "under_price": -115,
    "game_id": "2026_W02_CAR@ATL",
}

# An away favourite: Denver +2.5 at home, so Jacksonville is laid 2.5.
AWAY_FAV = {
    "home": "Denver Broncos",
    "away": "Jacksonville Jaguars",
    "home_ml": 120,
    "away_ml": -140,
    "home_point": 2.5,
    "home_spread_price": -104,
    "away_spread_price": -116,
    "total": 41.5,
    "over_price": -110,
    "under_price": -110,
    "game_id": "2026_W02_JAX@DEN",
}


# A week-3 Thursday game, so one payload can span two NFL weeks (T-33.2-02-13).
WEEK3_THURSDAY = {
    "home": "Miami Dolphins",
    "away": "Buffalo Bills",
    "home_ml": 110,
    "away_ml": -130,
    "home_point": 1.5,
    "home_spread_price": -110,
    "away_spread_price": -110,
    "total": 49.5,
    "over_price": -110,
    "under_price": -110,
    "game_id": "2026_W03_BUF@MIA",
}


def _schedule_row(game_id: str, home: str, away: str, kickoff: pd.Timestamp) -> dict:
    season, week = game_id.split("_", maxsplit=1)[0], game_id.split("_")[1]
    return {
        "game_id": game_id,
        "season": int(season),
        "week": int(week.lstrip("W")),
        "home_team": home,
        "away_team": away,
        "kickoff_et": kickoff,
    }


def _schedule(*, with_week3: bool = False) -> pd.DataFrame:
    """The silver ``games`` slice the ingest matches payload games against."""
    rows = [
        _schedule_row("2026_W02_CAR@ATL", "ATL", "CAR", WEEK2_SUNDAY_KICKOFF),
        _schedule_row("2026_W02_JAX@DEN", "DEN", "JAX", WEEK2_SUNDAY_KICKOFF),
    ]
    if with_week3:
        rows.append(
            _schedule_row("2026_W03_BUF@MIA", "MIA", "BUF", WEEK3_THURSDAY_KICKOFF)
        )
    return pd.DataFrame(rows)


def _locks(schedule: pd.DataFrame) -> dict[str, datetime]:
    """The lock map the ingest derives inside, rebuilt here through the ONE rule."""
    return {
        str(game_id): lock.to_pydatetime()
        for game_id, lock in lock_rule.lock_frame(schedule).items()
    }


def _event(
    fixture: dict,
    *,
    reverse_outcomes: bool = False,
    book: str = "draftkings",
    commence_time: str = "2026-09-20T17:00:00Z",
    last_update: str | None = "2026-09-15T04:00:00Z",
):
    """One event in the documented v4 shape. ``reverse_outcomes`` lists away first."""

    def ordered(pair: list[dict]) -> list[dict]:
        return list(reversed(pair)) if reverse_outcomes else pair

    bookmaker_times = {} if last_update is None else {"last_update": last_update}
    return {
        "id": fixture["game_id"],
        "commence_time": commence_time,
        "home_team": fixture["home"],
        "away_team": fixture["away"],
        "bookmakers": [
            {
                "key": book,
                **bookmaker_times,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": ordered(
                            [
                                {"name": fixture["home"], "price": fixture["home_ml"]},
                                {"name": fixture["away"], "price": fixture["away_ml"]},
                            ]
                        ),
                    },
                    {
                        "key": "spreads",
                        "outcomes": ordered(
                            [
                                {
                                    "name": fixture["home"],
                                    "price": fixture["home_spread_price"],
                                    "point": fixture["home_point"],
                                },
                                {
                                    "name": fixture["away"],
                                    "price": fixture["away_spread_price"],
                                    "point": -fixture["home_point"],
                                },
                            ]
                        ),
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {
                                "name": "Over",
                                "price": fixture["over_price"],
                                "point": fixture["total"],
                            },
                            {
                                "name": "Under",
                                "price": fixture["under_price"],
                                "point": fixture["total"],
                            },
                        ],
                    },
                ],
            }
        ],
    }


def _bare_ingester() -> OddsDataIngester:
    """The real ingester without its DB handle or HTTP client."""
    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.sportsbook_priority = []
    return ingester


def _transform(ingester: OddsDataIngester, raw: list[dict], schedule: pd.DataFrame):
    return ingester.transform_odds_data(
        raw, schedule=schedule, locks=_locks(schedule), captured_at=CAPTURED_AT
    )


def _validated_row(fixture: dict, *, reverse_outcomes: bool = False) -> pd.Series:
    """Drive one event through transform_odds_data AND validate_odds_data (OddsSchema)."""
    ingester = _bare_ingester()
    raw = [_event(fixture, reverse_outcomes=reverse_outcomes)]
    transformed = _transform(ingester, raw, _schedule())
    validated = ingester.validate_odds_data(transformed)
    assert len(validated) == 1, f"expected one validated row, got {len(validated)}"
    return validated.iloc[0]


BOTH_ORDERS = pytest.mark.parametrize(
    "reverse_outcomes", [False, True], ids=["home-listed-first", "away-listed-first"]
)
BOTH_FIXTURES = pytest.mark.parametrize(
    "fixture", [HOME_FAV, AWAY_FAV], ids=["home-favourite", "away-favourite"]
)


class TestMoneylinesMapByTeamName:
    """(a) ``ml_home`` / ``ml_away`` are populated, sided by name and never by position."""

    @BOTH_FIXTURES
    @BOTH_ORDERS
    def test_the_home_moneyline_is_the_home_teams_price(
        self, fixture, reverse_outcomes
    ):
        row = _validated_row(fixture, reverse_outcomes=reverse_outcomes)
        assert row["ml_home"] == fixture["home_ml"], (
            f"ml_home is {row['ml_home']!r}; {fixture['home']} priced "
            f"{fixture['home_ml']}. A blank or position-sided moneyline leaves WP unpriced."
        )

    @BOTH_FIXTURES
    @BOTH_ORDERS
    def test_the_away_moneyline_is_the_away_teams_price(
        self, fixture, reverse_outcomes
    ):
        row = _validated_row(fixture, reverse_outcomes=reverse_outcomes)
        assert row["ml_away"] == fixture["away_ml"], (
            f"ml_away is {row['ml_away']!r}; {fixture['away']} priced {fixture['away_ml']}"
        )


class TestTheSpreadIsPositiveWhenHomeIsFavoured:
    """(b) the stored convention: spread = -(home team's point)."""

    @BOTH_FIXTURES
    @BOTH_ORDERS
    def test_the_spread_is_the_negated_home_point(self, fixture, reverse_outcomes):
        row = _validated_row(fixture, reverse_outcomes=reverse_outcomes)
        expected = -fixture["home_point"]
        assert row["spread"] == expected, (
            f"{fixture['game_id']}: spread stored as {row['spread']!r}, expected "
            f"{expected} (positive = home favoured, the stored history's convention). "
            f"The home team's own line was {fixture['home_point']}."
        )


class TestTheSpreadJuicesLandOnTheirOwnSide:
    """(c) each spread price belongs to the side that carries it."""

    @BOTH_FIXTURES
    @BOTH_ORDERS
    def test_the_home_juice_is_the_home_price(self, fixture, reverse_outcomes):
        row = _validated_row(fixture, reverse_outcomes=reverse_outcomes)
        assert row["spread_ju_home"] == fixture["home_spread_price"], (
            f"{fixture['game_id']}: spread_ju_home is {row['spread_ju_home']!r}, but "
            f"{fixture['home']}'s spread price is {fixture['home_spread_price']}"
        )

    @BOTH_FIXTURES
    @BOTH_ORDERS
    def test_the_away_juice_is_the_away_price(self, fixture, reverse_outcomes):
        row = _validated_row(fixture, reverse_outcomes=reverse_outcomes)
        assert row["spread_ju_away"] == fixture["away_spread_price"], (
            f"{fixture['game_id']}: spread_ju_away is {row['spread_ju_away']!r}, but "
            f"{fixture['away']}'s spread price is {fixture['away_spread_price']}"
        )


class TestTotalsAreUnchanged:
    """(d) the totals branch was already right and must stay right."""

    @BOTH_FIXTURES
    def test_the_total_and_both_prices_are_stored(self, fixture):
        row = _validated_row(fixture)
        assert (row["total"], row["total_over_ju"], row["total_under_ju"]) == (
            fixture["total"],
            fixture["over_price"],
            fixture["under_price"],
        )


@pytest.fixture
def stored_through_the_real_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Both events through ``ingest_odds`` into a sandbox, read back as consumers read it."""

    class _StandIn:
        def get_nfl_odds(self, **_kwargs):
            return [_event(HOME_FAV), _event(AWAY_FAV, reverse_outcomes=True)]

        def close(self):
            return None

    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(tmp_path)))
    # Before the lock: a post-lock capture writes nothing (33.2 review C1 CR-02).
    monkeypatch.setattr(
        ingest_odds_module, "_observe_capture_instant", lambda: CAPTURED_AT
    )
    ingester = _bare_ingester()
    ingester.api_client = _StandIn()
    ingester.ingest_odds(season=2026, week=2, schedule=_schedule())
    stored = pd.read_parquet(tmp_path / "silver" / "odds_snapshot.parquet")
    return dedupe_odds_by_book_preference(stored).set_index("game_id")


class TestTheValuesSurviveTheRealStepPath:
    """A correct extractor must not be undone by the schema, the write or the dedupe."""

    @BOTH_FIXTURES
    def test_the_consumer_sees_the_extracted_values(
        self, stored_through_the_real_step, fixture
    ):
        row = stored_through_the_real_step.loc[fixture["game_id"]]
        observed = (
            row["ml_home"],
            row["ml_away"],
            row["spread"],
            row["spread_ju_home"],
            row["spread_ju_away"],
            row["total"],
        )
        expected = (
            fixture["home_ml"],
            fixture["away_ml"],
            -fixture["home_point"],
            fixture["home_spread_price"],
            fixture["away_spread_price"],
            fixture["total"],
        )
        assert observed == expected, (
            f"{fixture['game_id']} read back from silver/odds_snapshot.parquet after "
            f"dedupe as {observed}, expected {expected}"
        )


def _follows_home_favoured_rule(frame: pd.DataFrame) -> pd.Series:
    """True where a non-zero spread's sign agrees with the moneyline favourite."""
    home_favoured = frame["ml_home"] < frame["ml_away"]
    return (home_favoured & (frame["spread"] > 0)) | (
        ~home_favoured & (frame["spread"] < 0)
    )


class TestTheConventionIsTheStoredHistorys:
    """The sign rule lives in a test against REAL data, not only in prose."""

    def test_the_stored_history_uses_positive_equals_home_favoured(self):
        if not PRODUCTION_ODDS_PATH.exists():
            pytest.skip(PRODUCTION_ODDS_ABSENT_SKIP)
        stored = pd.read_parquet(
            PRODUCTION_ODDS_PATH, columns=["ml_home", "ml_away", "spread"]
        ).dropna()
        stored = stored[stored["spread"] != 0]
        assert len(stored) > 0, "no priced, non-pick'em rows to measure the rule on"
        share = float(_follows_home_favoured_rule(stored).mean())
        assert share >= STORED_CONVENTION_MINIMUM_SHARE, (
            f"only {share:.4f} of {len(stored)} stored rows follow positive = home "
            "favoured; the convention this extractor writes is no longer the history's"
        )

    @BOTH_FIXTURES
    def test_the_extractor_writes_the_same_rule(self, fixture):
        row = _validated_row(fixture)
        # The rule is only a statement about a PRICED row. Blank moneylines compare as
        # "not home favoured", which a negative spread would satisfy by accident -- so
        # populated moneylines are required before the sign is judged at all.
        assert pd.notna(row["ml_home"]) and pd.notna(row["ml_away"]), (
            f"{fixture['game_id']}: moneylines are blank (ml_home {row['ml_home']!r}, "
            f"ml_away {row['ml_away']!r}), so the stored rule cannot be checked"
        )
        frame = pd.DataFrame(
            [
                {
                    "ml_home": row["ml_home"],
                    "ml_away": row["ml_away"],
                    "spread": row["spread"],
                }
            ]
        )
        assert bool(_follows_home_favoured_rule(frame).iloc[0]), (
            f"{fixture['game_id']}: ml_home {row['ml_home']!r}, ml_away "
            f"{row['ml_away']!r}, spread {row['spread']!r} breaks the stored rule"
        )


# ---------------------------------------------------------------------------
# Plan 33.2-02 Task 2: the ONE snapshot_time is split into four named values.
# ---------------------------------------------------------------------------


def _row_for(frame: pd.DataFrame, game_id: str) -> pd.Series:
    rows = frame[frame["game_id"] == game_id]
    assert len(rows) == 1, f"expected one row for {game_id}, got {len(rows)}"
    return rows.iloc[0]


class TestEachRowCarriesItsOwnGamesLock:
    """``snapshot_ts`` is the game's own lock, matched through the schedule."""

    def test_a_two_week_payload_yields_two_weeks_and_two_locks_from_one_request(self):
        """T-33.2-02-13: the game id's week is the game's OWN week, not the request's.

        The request window spans eight days and so two NFL weeks. Before this plan every
        game in the payload was stamped with the requesting week and one instant.
        """
        ingester = _bare_ingester()
        raw = [
            _event(HOME_FAV),
            _event(WEEK3_THURSDAY, commence_time="2026-09-25T00:15:00Z"),
        ]
        transformed = _transform(ingester, raw, _schedule(with_week3=True))

        weeks = sorted({gid.split("_")[1] for gid in transformed["game_id"]})
        assert weeks == ["W02", "W03"], (
            f"a payload spanning two NFL weeks produced game-id weeks {weeks}; each game "
            "must carry its own schedule week, not the requesting week"
        )
        sunday = _row_for(transformed, HOME_FAV["game_id"])["snapshot_ts"]
        thursday = _row_for(transformed, WEEK3_THURSDAY["game_id"])["snapshot_ts"]
        assert sunday == WEEK2_SUNDAY_LOCK
        assert thursday == WEEK3_THURSDAY_LOCK
        assert sunday != thursday

    def test_the_same_two_week_split_survives_the_real_ingest_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """One ``ingest_odds`` call, one request, two locks stored."""

        class _StandIn:
            def get_nfl_odds(self, **_kwargs):
                return [
                    _event(HOME_FAV),
                    _event(WEEK3_THURSDAY, commence_time="2026-09-25T00:15:00Z"),
                ]

            def close(self):
                return None

        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )
        # A capture before both locks: a post-lock capture writes nothing (CR-02).
        monkeypatch.setattr(
            ingest_odds_module, "_observe_capture_instant", lambda: CAPTURED_AT
        )
        ingester = _bare_ingester()
        ingester.api_client = _StandIn()
        validated = ingester.ingest_odds(
            season=2026, week=2, schedule=_schedule(with_week3=True)
        )

        assert validated["snapshot_ts"].nunique() == 2
        assert set(validated["game_id"]) == {
            HOME_FAV["game_id"],
            WEEK3_THURSDAY["game_id"],
        }

    def test_the_lock_is_the_one_rules_answer_for_that_kickoff(self):
        ingester = _bare_ingester()
        transformed = _transform(ingester, [_event(AWAY_FAV)], _schedule())

        row = _row_for(transformed, AWAY_FAV["game_id"])
        assert row["snapshot_ts"] == lock_rule.game_lock(WEEK2_SUNDAY_KICKOFF)


class TestAnAbsentBookmakerTimeIsNull:
    """T-33.2-02-12: an unknown upstream time is NULL, never our own instant."""

    @pytest.mark.parametrize(
        "last_update", [None, "not-a-timestamp"], ids=["absent", "unparseable"]
    )
    def test_last_update_is_null_and_differs_from_capture_and_lock(self, last_update):
        ingester = _bare_ingester()
        transformed = _transform(
            ingester, [_event(HOME_FAV, last_update=last_update)], _schedule()
        )
        validated = ingester.validate_odds_data(transformed)
        row = validated.iloc[0]

        assert row["last_update"] is None or pd.isna(row["last_update"]), (
            f"an {last_update!r} bookmaker time was stored as {row['last_update']!r}. "
            "Filling an unknown upstream time with our own instant makes a manufactured "
            "stamp read as the bookmaker's (RESEARCH P1)."
        )
        assert row["last_update"] != row["created_at"]
        assert row["last_update"] != row["snapshot_ts"]

    def test_a_present_bookmaker_time_is_kept_as_the_upstream_time(self):
        ingester = _bare_ingester()
        transformed = _transform(ingester, [_event(HOME_FAV)], _schedule())

        assert _row_for(transformed, HOME_FAV["game_id"])["last_update"] == datetime(
            2026, 9, 15, 4, 0, tzinfo=UTC
        )


class TestTheCaptureInstantIsObservedNotSupplied:
    """``created_at`` is the instant the response was OBSERVED inside the ingest."""

    def test_every_row_of_one_response_shares_the_observed_instant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        class _StandIn:
            def get_nfl_odds(self, **_kwargs):
                return [_event(HOME_FAV), _event(AWAY_FAV, book="fanduel")]

            def close(self):
                return None

        observed: list[datetime] = []

        def _clock() -> datetime:
            observed.append(CAPTURED_AT)
            return CAPTURED_AT

        monkeypatch.setattr(ingest_odds_module, "_observe_capture_instant", _clock)
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )
        ingester = _bare_ingester()
        ingester.api_client = _StandIn()
        validated = ingester.ingest_odds(season=2026, week=2, schedule=_schedule())

        assert len(observed) == 1, (
            f"the capture clock was read {len(observed)} times; one response has ONE "
            "capture instant, read once when it returns"
        )
        assert len(validated) == 2
        assert set(validated["created_at"]) == {CAPTURED_AT}, (
            "created_at is not the observed capture instant -- a second clock read at "
            "the write would stamp a different time from the one the data was captured at"
        )

    def test_the_public_surface_takes_no_capture_instant_or_lock_map(self):
        import inspect

        params = set(inspect.signature(OddsDataIngester.ingest_odds).parameters)
        assert {"schedule", "commence_from", "commence_to"} <= params
        assert not ({"captured_at", "locks", "snapshot_time"} & params)


class TestPayloadGamesAreMatchedToTheSchedule:
    """T-33.2-02-11: the schedule, not the payload, says when a game starts."""

    def test_an_unmatched_payload_game_is_skipped_and_named(self):
        ingester = _bare_ingester()
        stray = dict(WEEK3_THURSDAY)
        transformed = _transform(
            ingester,
            [_event(HOME_FAV), _event(stray, commence_time="2026-09-25T00:15:00Z")],
            _schedule(),  # no week-3 row
        )

        assert set(transformed["game_id"]) == {HOME_FAV["game_id"]}
        report = ingester.last_match_report
        assert any(
            "Buffalo Bills" in name and "Miami Dolphins" in name
            for name in report.unmatched_games
        ), f"the unmatched game is not named in {report.unmatched_games}"

    def test_a_commence_time_disagreement_is_reported_and_the_schedule_wins(self):
        ingester = _bare_ingester()
        moved = "2026-09-20T20:25:00Z"  # 3h25m after the scheduled 1 PM ET kickoff
        transformed = _transform(
            ingester, [_event(HOME_FAV, commence_time=moved)], _schedule()
        )

        assert ingester.last_match_report.kickoff_disagreements == (
            HOME_FAV["game_id"],
        )
        assert _row_for(transformed, HOME_FAV["game_id"])[
            "snapshot_ts"
        ] == lock_rule.game_lock(WEEK2_SUNDAY_KICKOFF)

    def test_an_agreeing_commence_time_reports_nothing(self):
        ingester = _bare_ingester()
        _transform(ingester, [_event(HOME_FAV)], _schedule())

        assert ingester.last_match_report.kickoff_disagreements == ()
        assert ingester.last_match_report.unmatched_games == ()


class TestEveryTimestampHandedToTheSchemaIsAware:
    """The schema's naive-relabel branch (data/schemas.py) is unreachable from here."""

    def test_no_naive_timestamp_reaches_odds_schema(self):
        ingester = _bare_ingester()
        transformed = _transform(
            ingester,
            [_event(HOME_FAV), _event(AWAY_FAV, last_update=None)],
            _schedule(),
        )

        for column in ("snapshot_ts", "created_at", "last_update"):
            for value in transformed[column]:
                if value is None or pd.isna(value):
                    continue
                assert pd.Timestamp(value).tzinfo is not None, (
                    f"{column} value {value!r} is naive; OddsSchema would RELABEL it as "
                    "UTC rather than refuse it"
                )


class TestThePipelineStepPassesTheScheduleExplicitly:
    """``step_ingest_odds`` still takes no arguments and hands the ingest its slate."""

    def test_the_step_loads_the_slate_and_passes_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from pipeline import steps

        calls: dict[str, object] = {}
        slate = _schedule()

        class _Recorder:
            def ingest_odds(self, **kwargs):
                calls.update(kwargs)
                return pd.DataFrame()

        monkeypatch.setattr(steps, "_resolve_current_week", lambda: (2026, 2))
        monkeypatch.setattr(ingest_odds_module, "OddsDataIngester", _Recorder)
        monkeypatch.setattr(
            ingest_odds_module,
            "load_schedule_slice",
            lambda season, week: slate if (season, week) == (2026, 2) else None,
        )

        steps.step_ingest_odds()

        assert calls.get("season") == 2026
        assert calls.get("week") == 2
        assert calls.get("schedule") is slate, (
            "the step did not pass the loaded slate as `schedule`; the ingest would have "
            "no schedule to derive its per-game locks from"
        )
        assert "locks" not in calls, "the step must not derive a second lock map"


class TestACaptureAfterTheLockIsNeverWritten:
    """33.2 review C1 CR-02: a post-lock (or in-game) capture is refused by name.

    Every live row is stamped ``snapshot_ts = lock``. Nothing compared the capture instant with
    that lock, so a Sunday-afternoon run wrote post-lock and IN-PLAY lines labelled as known at
    Saturday's lock, and every ``snapshot_ts <= lock`` fence admitted them. A game whose lock
    has passed at the capture instant is now left out and named; at-lock is still admissible.
    """

    def _transform_at(self, ingester: OddsDataIngester, captured_at: datetime):
        schedule = _schedule()
        return ingester.transform_odds_data(
            [_event(HOME_FAV)],
            schedule=schedule,
            locks=_locks(schedule),
            captured_at=captured_at,
        )

    def test_a_capture_one_second_after_the_lock_writes_no_row_and_names_the_game(self):
        ingester = _bare_ingester()
        late = WEEK2_SUNDAY_LOCK.astimezone(UTC) + timedelta(seconds=1)
        transformed = self._transform_at(ingester, late)

        assert transformed.empty, "a post-lock capture was written under the lock label"
        assert ingester.last_match_report.post_lock_games == (HOME_FAV["game_id"],)

    def test_a_capture_exactly_at_the_lock_is_written_with_its_real_capture_time(self):
        ingester = _bare_ingester()
        at_lock = WEEK2_SUNDAY_LOCK.astimezone(UTC)
        transformed = self._transform_at(ingester, at_lock)

        row = _row_for(transformed, HOME_FAV["game_id"])
        assert row["created_at"] == at_lock
        assert ingester.last_match_report.post_lock_games == ()
