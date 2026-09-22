"""STANDING GUARD: no 2002-2025 kickoff before noon Eastern unless it is a game abroad.

Plan 33.2-12, p332_ extra step 3c (orchestrator-assigned; deferred-items entry found by
Plan 33.2-10).

THE DEFECT THIS GUARDS
----------------------
Every 2002-2005 Monday and Thursday NIGHT game (68 games) was stored at 09:00 ET: the feed's
``gametime`` is a 12-hour AM/PM error, and NFL.com carries the same value. The date was
right, so no lock moved and nothing noticed, but every reader of the kickoff HOUR (rest-day
counts, the day-before forecast hour) read a morning kickoff for a night game.

THE ONE LEGITIMATE MORNING KICKOFF, AS A RULE, NOT A LIST
---------------------------------------------------------
The NFL's games abroad kick off at 9:30 AM ET. ``config/kickoff_hour_corrections.toml``
carries that as a cited ``[international_morning_exception]`` rule -- a Sunday, a venue whose
``data/venues.json`` country is not the USA, 09:30 ET -- so a new game abroad needs no edit
here, and a US game at 09:30 still fails.

THE 68 CORRECTED GAMES ARE ASSERTED, NOT EXEMPTED
-------------------------------------------------
Silver must carry, for every game of 2002-2005, exactly the kickoff a real re-ingest of the
PINNED feed writes through ``GameDataIngester.transform_schedule_data`` -- the path that
applies ``resolve_kickoff_gametime``. A store and an ingest that disagree fail here.

READ-ONLY. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from conf.season_partition import CORPUS_FIRST_SEASON
from scripts import ingest_games

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"
VENUES_PATH = REPO_ROOT / "data" / "venues.json"
RECORD_PATH = REPO_ROOT / "config" / "kickoff_hour_corrections.toml"

# The corpus floor, read from the one rule module (Plan 33.2-18: no floor literal outside
# conf/season_partition.py). Was: FIRST_SEASON = 2002.
FIRST_SEASON = CORPUS_FIRST_SEASON
LAST_SEASON = 2025
CORRECTED_SEASONS = (2002, 2003, 2004, 2005)
NOON_MINUTES = 12 * 60
ET = "America/New_York"


def _silver_games() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        pytest.skip("silver games is not built on this checkout")
    games = pd.read_parquet(GAMES_PATH)
    return games[games["season"].between(FIRST_SEASON, LAST_SEASON)].reset_index(
        drop=True
    )


def _exception_rule() -> dict[str, str]:
    return tomllib.loads(RECORD_PATH.read_text(encoding="utf-8"))[
        "international_morning_exception"
    ]


def _venue_countries() -> dict[str, str]:
    venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
    return {str(v["stadium_id"]): str(v["country"]) for v in venues}


def morning_kickoffs_outside_the_rule(games: pd.DataFrame) -> list[str]:
    """Game ids kicking off before noon ET that the international rule does not admit."""
    rule = _exception_rule()
    countries = _venue_countries()
    local = games["kickoff_et"].dt.tz_convert(ET)
    minutes = local.dt.hour * 60 + local.dt.minute
    morning = games.loc[minutes < NOON_MINUTES]
    offenders = []
    for position, game in morning.iterrows():
        wall = local.loc[position]
        admitted = (
            wall.day_name() == rule["weekday"]
            and wall.strftime("%H:%M") == rule["kickoff_et"]
            and countries.get(str(game["stadium_id"]), "USA") != "USA"
        )
        if not admitted:
            offenders.append(str(game["game_id"]))
    return offenders


class TestNoMorningKickoffUnlessAbroad:
    def test_no_2002_2025_kickoff_is_before_noon_et_outside_the_rule(self) -> None:
        offenders = morning_kickoffs_outside_the_rule(_silver_games())
        assert offenders == [], (
            f"{len(offenders)} 2002-2025 game(s) kick off before noon ET and are not a "
            f"Sunday 09:30 game abroad, e.g. {offenders[:5]}. A night game stored as a "
            "morning one is the 12-hour AM/PM defect; correct it with a cited entry in "
            "config/kickoff_hour_corrections.toml, never by widening the rule."
        )

    def test_the_rule_admits_the_games_abroad_non_vacuously(self) -> None:
        """The exception is exercised: morning games exist and every one is abroad."""
        games = _silver_games()
        local = games["kickoff_et"].dt.tz_convert(ET)
        morning = games.loc[local.dt.hour < 12]
        countries = _venue_countries()
        assert len(morning) > 0, "no morning kickoff at all -- the rule is untested"
        assert all(countries[str(s)] != "USA" for s in morning["stadium_id"])

    def test_a_planted_us_night_game_at_nine_am_is_flagged(self) -> None:
        games = _silver_games()
        planted = games[
            (games["season"] == 2010) & (games["stadium_id"] == "GNB00")
        ].head(1)
        assert len(planted) == 1
        planted = planted.assign(
            kickoff_et=pd.Timestamp("2010-11-15 09:00", tz=ET).tz_convert("UTC")
        )
        assert morning_kickoffs_outside_the_rule(planted) == list(planted["game_id"])

    def test_a_planted_us_game_at_nine_thirty_on_sunday_is_still_flagged(self) -> None:
        """The rule is the venue's country, not the clock: 09:30 at Lambeau fails."""
        games = _silver_games()
        planted = games[
            (games["season"] == 2010) & (games["stadium_id"] == "GNB00")
        ].head(1)
        planted = planted.assign(
            kickoff_et=pd.Timestamp("2010-11-14 09:30", tz=ET).tz_convert("UTC")
        )
        assert morning_kickoffs_outside_the_rule(planted) == list(planted["game_id"])


class TestTheStoreEqualsARealReingest:
    """Silver 2002-2005 kickoffs are what the ingest writes from the pinned feed."""

    @pytest.fixture(scope="class")
    def reingested(self) -> pd.DataFrame:
        from data import upstream_pin

        feed = upstream_pin.load_schedules(list(CORRECTED_SEASONS))
        feed = feed.to_pandas() if hasattr(feed, "to_pandas") else feed
        out = ingest_games.GameDataIngester().transform_schedule_data(feed)
        kickoffs = pd.to_datetime(out["kickoff_et"]).dt.tz_localize(ET)
        return out.assign(kickoff_et=kickoffs.dt.tz_convert("UTC"))

    def test_every_2002_2005_kickoff_matches_the_reingest(
        self, reingested: pd.DataFrame
    ) -> None:
        silver = _silver_games()
        silver = silver[silver["season"].isin(CORRECTED_SEASONS)]
        joined = silver.merge(
            reingested[["game_id", "kickoff_et"]],
            on="game_id",
            suffixes=("", "_ingest"),
        )
        assert len(joined) == len(silver) == len(reingested)
        mismatched = joined.loc[
            joined["kickoff_et"] != joined["kickoff_et_ingest"], "game_id"
        ].tolist()
        assert mismatched == [], f"store and ingest disagree on {mismatched[:5]}"

    def test_the_reingest_applies_every_recorded_correction(
        self, reingested: pd.DataFrame
    ) -> None:
        corrections = ingest_games.load_kickoff_hour_corrections()
        local = reingested.set_index("game_id")["kickoff_et"].dt.tz_convert(ET)
        assert len(corrections) == 68
        for game_id, correction in corrections.items():
            assert local[game_id].strftime("%Y-%m-%d %H:%M") == (
                f"{correction.gameday} {correction.corrected_gametime}"
            ), game_id

    def test_the_pinned_feed_still_carries_the_defect(self) -> None:
        """The override is live, not a no-op: the pinned feed reads 09:00 for all 68."""
        from data import upstream_pin

        feed = upstream_pin.load_schedules(list(CORRECTED_SEASONS))
        feed = feed.to_pandas() if hasattr(feed, "to_pandas") else feed
        morning = feed[(feed["gametime"] == "09:00")]
        assert len(morning) == len(ingest_games.load_kickoff_hour_corrections())
