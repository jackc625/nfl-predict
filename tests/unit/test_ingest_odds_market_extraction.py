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

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import data.storage as storage_mod
from backtest.ou_divergence import dedupe_odds_by_book_preference
from data.storage import ParquetManager
from scripts.ingest_odds import OddsDataIngester

ET = ZoneInfo("America/New_York")
SNAPSHOT_TIME = datetime(2026, 9, 18, 18, 0, tzinfo=ET)

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


def _event(fixture: dict, *, reverse_outcomes: bool = False, book: str = "draftkings"):
    """One event in the documented v4 shape. ``reverse_outcomes`` lists away first."""

    def ordered(pair: list[dict]) -> list[dict]:
        return list(reversed(pair)) if reverse_outcomes else pair

    return {
        "id": fixture["game_id"],
        "commence_time": "2026-09-20T17:00:00Z",
        "home_team": fixture["home"],
        "away_team": fixture["away"],
        "bookmakers": [
            {
                "key": book,
                "last_update": "2026-09-15T04:00:00Z",
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


def _validated_row(fixture: dict, *, reverse_outcomes: bool = False) -> pd.Series:
    """Drive one event through transform_odds_data AND validate_odds_data (OddsSchema)."""
    ingester = _bare_ingester()
    raw = [_event(fixture, reverse_outcomes=reverse_outcomes)]
    transformed = ingester.transform_odds_data(raw, SNAPSHOT_TIME, 2026, 2)
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
    ingester = _bare_ingester()
    ingester.api_client = _StandIn()
    ingester.ingest_odds(season=2026, week=2, snapshot_time=SNAPSHOT_TIME)
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
