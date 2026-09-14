"""The routing change, MEASURED per game against the real prior rule (D33.1-09).

Phase 33.1, Plan 33.1-03 Task 2 (R1, D33.1-06, D33.1-09, T-33.1-15/18/19/20).

WHY A COORDINATE DIFF AND NOT A WEATHER-VALUE DIFF
----------------------------------------------------
D33.1-09: comparing the coordinates the OLD home-team rule resolves against the
coordinates the NEW ``stadium_id`` rule resolves is EXACT, UNCONFOUNDED and CHEAP.
A weather-value diff over the same games is confounded, because D33.1-08's
venue-local-hour fix lands in the same phase and moves a value on essentially
every comparable row. A number that moved for two reasons at once cannot be
attributed to either.

THE OLD RULE IS CALLED, NOT REBUILT
------------------------------------
``WeatherDataIngester._get_venue_record(home_team, venues_df)`` is the real prior
lookup and Plan 33.1-03 Task 1 deliberately left it in the module. This module
calls it. A paraphrase of the old rule would make this a comparison between the
new code and somebody's memory of the old code, and the thing most worth catching
is precisely a prior behaviour that was not what anyone remembered. The result is
MEMOIZED per home-team abbreviation -- that is a cache over the real call, not a
second implementation of it.

NEUTRAL-SITE GAMES COME FROM THE FEED'S OWN `location`, NEVER FROM SILVER
--------------------------------------------------------------------------
``data/silver/games.parquet`` carries ``neutral_site`` and every one of its 6,499
cells reads False, because ``row.get("neutral_site", False)`` never reads a feed
value at all -- the bronze schedule has no such column
(``tests/phase33_state.py``, the Plan 33-06 identity-columns slot). Phase 33 Wave
12 backfills it and has not run. Reading that column here would report ZERO
neutral-site games and the repair would look vacuous.

THE MEASUREMENT IS RE-DERIVED HERE, NEVER TRANSCRIBED
------------------------------------------------------
Every figure is computed from the pinned feed at test time and compared against
``tests.phase33_state.ROUTING_COORDINATE_DIFF``, which recorded the same
computation on 2026-09-13. That is the T-33.1-18 mitigation: a constant that only
ever agrees with itself measures nothing.

THIS MODULE READS AND NEVER WRITES.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.upstream_pin import load_schedules
from features import contextual
from scripts import ingest_weather
from tests import phase33_state

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

FIRST_SEASON = 2002
LAST_SEASON = 2025

# R4's rule for "weather applies to this game", read from the feed's own per-game
# roof rather than from the venue's roof_type.
WEATHER_APPLICABLE_ROOFS = frozenset({"outdoors", "open"})

# The nflverse marker for a neutral-site game, in the feed's own column.
NEUTRAL_LOCATION = "Neutral"

# The twelve largest misroutes and the successor each one used to resolve to,
# named in the plan's acceptance criteria. Asserted in BOTH directions below, so
# each row proves a CHANGE rather than a state.
#
# `LON00` (26 games) is larger than `LAX97` (22) and `MIN98` (18) and is
# deliberately NOT in this list: it is a neutral-site venue whose old answers
# scatter across ELEVEN different successors, so "the successor it used to resolve
# to" is not a well-formed claim about it. It is present in the recorded
# per-stadium breakdown with all eleven.
NAMED_MISROUTES: tuple[tuple[str, str, int], ...] = (
    ("OAK00", "VEG00", 141),
    ("NYC00", "NYC01", 132),
    ("SDG00", "LAX01", 125),
    ("ATL00", "ATL97", 125),
    ("STL00", "LAX01", 112),
    ("SFO00", "SFO01", 99),
    ("MIN00", "MIN01", 95),
    ("DAL99", "DAL00", 57),
    ("IND99", "IND00", 54),
    ("PHO99", "PHO00", 32),
    ("LAX99", "LAX01", 31),
    ("LAX97", "LAX01", 22),
)


@lru_cache(maxsize=1)
def _schedules() -> pd.DataFrame:
    return load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))


@lru_cache(maxsize=1)
def _venue_records() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"])


@lru_cache(maxsize=1)
def _venues_frame() -> pd.DataFrame:
    return pd.DataFrame(list(_venue_records()))


@lru_cache(maxsize=1)
def _old_rule_by_home_team() -> dict[str, dict[str, Any]]:
    """The OLD home-team resolution, evaluated once per distinct abbreviation.

    ``_get_venue_record`` is the REAL prior lookup, called here rather than
    reimagined. It depends only on the home team, so one call per abbreviation is
    a cache and not a paraphrase: there are roughly 35 distinct abbreviations
    across 6,499 games, and calling it per game would pay a 60-row pandas scan
    6,499 times for an answer that cannot differ.
    """
    ingester = ingest_weather.WeatherDataIngester()
    venues_df = _venues_frame()
    return {
        team: ingester._get_venue_record(team, venues_df)
        for team in sorted({str(t) for t in _schedules()["home_team"]})
    }


@lru_cache(maxsize=1)
def _measure() -> dict[str, Any]:
    """Re-derive the whole diff from the pinned feed. Read only."""
    records = list(_venue_records())
    old_by_team = _old_rule_by_home_team()
    historical_ids = set(phase33_state.HISTORICAL_STADIUM_IDS)

    per_stadium: dict[str, dict[str, Any]] = {}
    games_changed = 0
    naming_new = 0
    neutral_changed = 0
    weather_changed = 0
    neutral_total = 0
    neutral_resolving_own = 0

    for row in _schedules().itertuples():
        stadium_id = str(row.stadium_id)
        game = {"game_id": str(row.game_id), "stadium_id": stadium_id}
        new = contextual.resolve_venue_for_game(game, records)
        old = old_by_team[str(row.home_team)]
        is_neutral = str(row.location) == NEUTRAL_LOCATION

        if is_neutral:
            neutral_total += 1
            neutral_resolving_own += int(str(new["stadium_id"]) == stadium_id)

        if (old["latitude"], old["longitude"]) == (new["latitude"], new["longitude"]):
            continue

        games_changed += 1
        naming_new += int(stadium_id in historical_ids)
        neutral_changed += int(is_neutral)
        weather_applicable = str(row.roof) in WEATHER_APPLICABLE_ROOFS
        weather_changed += int(weather_applicable)

        entry = per_stadium.setdefault(
            stadium_id,
            {"games": 0, "successors": Counter(), "neutral": 0, "weather": 0},
        )
        entry["games"] += 1
        entry["successors"][str(old["stadium_id"])] += 1
        entry["neutral"] += int(is_neutral)
        entry["weather"] += int(weather_applicable)

    breakdown: dict[str, dict[str, Any]] = {}
    for code, entry in per_stadium.items():
        successors: Counter = entry["successors"]
        # The plurality successor, with ties broken ALPHABETICALLY rather than by
        # `Counter.most_common`, whose tie order is insertion order -- that is, the
        # order games happen to appear in the feed. Several neutral-site venues
        # have ten or eleven successors at one game each, so an insertion-ordered
        # answer would be a fact about the schedule's sort rather than about the
        # routing, and would flip the recorded value if the pin were ever re-sorted.
        primary, primary_games = min(successors.items(), key=lambda kv: (-kv[1], kv[0]))
        breakdown[code] = {
            "games": entry["games"],
            "successor": primary,
            "successor_games": primary_games,
            "successors": dict(sorted(successors.items())),
            "neutral_site_games": entry["neutral"],
            "weather_applicable_games": entry["weather"],
        }

    return {
        "games_total": len(_schedules()),
        "games_changed": games_changed,
        "distinct_stadium_ids_changed": len(breakdown),
        "games_naming_a_newly_added_id": naming_new,
        "games_naming_an_already_known_id": games_changed - naming_new,
        "neutral_site_games": neutral_changed,
        "weather_applicable_games": weather_changed,
        "per_stadium_id": breakdown,
        "neutral_site_games_total": neutral_total,
        "neutral_site_games_resolving_to_their_own_stadium": neutral_resolving_own,
    }


def _recorded() -> dict[str, Any]:
    """The committed measurement, read at CALL time so its absence is a RED test."""
    return phase33_state.ROUTING_COORDINATE_DIFF  # type: ignore[attr-defined]


def _recorded_repair() -> dict[str, Any]:
    return phase33_state.HISTORICAL_NEUTRAL_MISRESOLUTION_REPAIRED  # type: ignore[attr-defined]


class TestTheDiffIsNonEmptyAndReproducesTheRecord:
    """Test 2 -- the measurement, re-derived rather than transcribed."""

    def test_the_changed_set_is_not_empty(self) -> None:
        """Non-vacuity: a diff of zero would make every assertion below trivial."""
        assert _measure()["games_changed"] > 0

    @pytest.mark.parametrize(
        "field",
        [
            "games_total",
            "games_changed",
            "distinct_stadium_ids_changed",
            "games_naming_a_newly_added_id",
            "games_naming_an_already_known_id",
            "neutral_site_games",
            "weather_applicable_games",
        ],
    )
    def test_each_recorded_scalar_reproduces(self, field: str) -> None:
        measured = _measure()[field]
        recorded = _recorded()[field]
        assert measured == recorded, (
            f"{field} re-derives to {measured!r} against the recorded {recorded!r}. "
            "The recorded value was measured on 2026-09-13 from the same pinned "
            "feed; a divergence means the pin, the venue table or the routing rule "
            "moved. Record BOTH numbers rather than reconciling to the expected one."
        )

    def test_the_split_closes(self) -> None:
        """Arithmetic, checked rather than trusted: the two halves sum."""
        measured = _measure()
        assert (
            measured["games_naming_a_newly_added_id"]
            + measured["games_naming_an_already_known_id"]
            == measured["games_changed"]
        )

    def test_the_whole_per_stadium_breakdown_reproduces(self) -> None:
        measured = _measure()["per_stadium_id"]
        recorded = _recorded()["per_stadium_id"]
        assert set(measured) == set(recorded), (
            "the set of stadium_ids whose games changed venue is not the recorded "
            f"set. Only in the measurement: {sorted(set(measured) - set(recorded))}; "
            f"only in the record: {sorted(set(recorded) - set(measured))}."
        )
        for code in sorted(measured):
            assert measured[code] == recorded[code], (
                f"{code} re-derives to {measured[code]!r} against the recorded "
                f"{recorded[code]!r}."
            )

    def test_the_breakdown_games_sum_to_the_total(self) -> None:
        measured = _measure()
        assert (
            sum(entry["games"] for entry in measured["per_stadium_id"].values())
            == measured["games_changed"]
        )


class TestTheTwelveLargestMisroutes:
    """Test 3 -- each named pair, asserted in BOTH directions.

    Asserting only that the NEW rule is right would pass on a tree where the old
    rule had also been right, which would mean the diff measured nothing. Each row
    therefore asserts what the OLD rule answered as well.
    """

    @pytest.mark.parametrize(
        ("code", "successor", "games"),
        NAMED_MISROUTES,
        ids=[code for code, _, _ in NAMED_MISROUTES],
    )
    def test_the_named_pair_changed_by_the_recorded_count(
        self, code: str, successor: str, games: int
    ) -> None:
        entry = _measure()["per_stadium_id"].get(code)
        assert entry is not None, (
            f"{code} is absent from the changed set entirely, so its games now "
            "resolve the same way they did before. That is what half-retiring the "
            "two-condition gate looks like."
        )
        assert entry["games"] == games, (
            f"{code} moved {entry['games']} games, expected {games}."
        )
        assert entry["successor"] == successor, (
            f"{code} used to resolve to {entry['successor']!r}, not to "
            f"{successor!r}. The recorded successors are "
            f"{entry['successors']!r}."
        )

    @pytest.mark.parametrize(
        ("code", "successor", "games"),
        NAMED_MISROUTES,
        ids=[code for code, _, _ in NAMED_MISROUTES],
    )
    def test_every_game_now_resolves_to_its_own_stadium(
        self, code: str, successor: str, games: int
    ) -> None:
        """The NEW direction, per game rather than in aggregate."""
        records = list(_venue_records())
        rows = [r for r in _schedules().itertuples() if str(r.stadium_id) == code]
        assert rows, f"no pinned game names {code}"
        for row in rows:
            venue = contextual.resolve_venue_for_game(
                {"game_id": str(row.game_id), "stadium_id": code}, records
            )
            assert str(venue["stadium_id"]) == code
            assert str(venue["stadium_id"]) != successor


class TestTheOldRuleIsStillWrong:
    """Test 4 -- non-vacuity for the whole module.

    If the old rule had quietly started giving the right answer -- a venue record
    edited, an abbreviation remapped -- the diff above would still compute, would
    report zero, and every "changed" assertion would be measuring a rule against
    itself. This asserts the old rule REMAINS wrong for named games, so the diff
    is a comparison between two different answers.
    """

    def test_a_named_oakland_game_still_resolves_to_las_vegas_under_the_old_rule(
        self,
    ) -> None:
        old = _old_rule_by_home_team()["OAK"]
        assert str(old["stadium_id"]) == "VEG00", (
            "the OLD home-team rule no longer answers Allegiant for an Oakland "
            "home game, so this module is comparing the new rule against itself."
        )

    def test_the_old_and_new_answers_are_hundreds_of_miles_apart(self) -> None:
        """The size of the error, not just its existence."""
        calculator = contextual.ContextualFeaturesCalculator()
        records = {str(r["stadium_id"]): r for r in _venue_records()}
        old = _old_rule_by_home_team()["OAK"]
        new = records["OAK00"]
        miles = calculator._calculate_distance(
            old["latitude"], old["longitude"], new["latitude"], new["longitude"]
        )
        assert miles > 300, (
            f"Oakland Coliseum and the stadium the old rule chose are {miles:.0f} "
            "miles apart. If that number collapses, the venue table moved."
        )

    def test_the_old_rule_is_wrong_for_more_than_one_franchise(self) -> None:
        old_by_team = _old_rule_by_home_team()
        for team, expected_successor in (
            ("OAK", "VEG00"),
            ("SD", "LAX01"),
            ("STL", "LAX01"),
        ):
            assert str(old_by_team[team]["stadium_id"]) == expected_successor, (
                f"the old rule answers {old_by_team[team]['stadium_id']!r} for "
                f"{team}, not {expected_successor!r}."
            )


class TestTheNeutralSiteDisclosureIsRepaired:
    """Test 5 -- the 91 games D33-15 deferred, closed by D33.1-06."""

    def test_the_feed_still_marks_the_recorded_number_of_neutral_games(self) -> None:
        """Read from the FEED's `location`, never from silver's `neutral_site`."""
        measured = _measure()["neutral_site_games_total"]
        recorded = phase33_state.HISTORICAL_NEUTRAL_MISRESOLUTION["games"]
        assert measured == recorded, (
            f"the pinned 2002-2025 feed marks {measured} games location == "
            f"'Neutral', against the {recorded} the disclosure recorded. The "
            "repair below is scoped to exactly that population."
        )

    def test_silver_now_agrees_with_the_feed_and_the_source_stays_the_feed(
        self,
    ) -> None:
        """THE WAVE-12 DECISION, recorded where the tripwire asked for it.

        This test was `test_silver_would_have_reported_zero`. It asserted that
        silver's `neutral_site` was a constant False, so that when Phase 33 Wave 12
        backfilled the column the test would turn RED and somebody would decide
        whether this module should switch its source -- rather than the source
        silently becoming viable while the comment above still said it was not.

        WAVE 12 HAS RUN (Plan 33-12). THE DECISION IS: THE SOURCE STAYS THE FEED.

        Two reasons, and the first is the binding one.

        1. Silver's `neutral_site` is DERIVED from the feed's `location` by
           `scripts.ingest_games._derive_neutral_site`. Measuring the neutral-site
           population from silver would therefore be measuring the derivation
           against itself, and this module's whole job is to state a disclosure
           about the PINNED UPSTREAM population -- which has to stay anchored on
           the immutable bytes, not on a store any later plan may rewrite.
        2. The disclosure is a historical record. Re-sourcing it would make the
           numbers move whenever silver is re-ingested, which is the opposite of
           what a record is for.

        WHAT IS ASSERTED INSTEAD IS STRICTLY STRONGER than the old constant-False
        claim: silver and the feed now AGREE, game for game. That was not
        assertable before the backfill, because one side was a constant.
        """
        silver_path = REPO_ROOT / "data" / "silver" / "games.parquet"
        if not silver_path.is_file():
            pytest.skip("silver games.parquet is absent from this checkout")
        silver = pd.read_parquet(silver_path, columns=None)
        if "neutral_site" not in silver.columns:
            pytest.skip("silver games carries no neutral_site column yet")
        if not silver["neutral_site"].any():
            pytest.skip(
                "silver's neutral_site is still a constant False, so the Plan "
                "33-12 backfill has not run against this checkout. The agreement "
                "below is only meaningful once it has."
            )

        # COMPARED PER SEASON, NOT PER game_id, and the reason is a real
        # incompatibility rather than convenience: the FEED keys games as
        # `2002_21_OAK_TB` and silver keys them as `2002_W21_LV@TB` -- a different
        # scheme AND a different team vocabulary (the feed's OAK against the
        # canonical LV). Bridging the two here would mean re-implementing
        # `_create_game_id` inside the test, which is the fixture-agrees-with-itself
        # failure this module exists to avoid. The per-season histogram is
        # instrument-independent and, at 91 games over 24 seasons, tight.
        feed_by_season = Counter(
            int(row.season)
            for row in _schedules().itertuples()
            if str(row.location) == NEUTRAL_LOCATION
        )
        silver_by_season = Counter(
            int(season) for season in silver.loc[silver["neutral_site"], "season"]
        )

        assert silver_by_season == feed_by_season, (
            "silver's neutral_site and the feed's `location` disagree season by "
            f"season: silver {dict(sorted(silver_by_season.items()))} against feed "
            f"{dict(sorted(feed_by_season.items()))}. The column is a derivation of "
            "the feed, so a disagreement means a row was written by something "
            "other than _derive_neutral_site -- and this module would report a "
            "different population depending on which side it read."
        )
        assert (
            sum(feed_by_season.values())
            == phase33_state.HISTORICAL_NEUTRAL_MISRESOLUTION["games"]
        ), "the agreed population must still be the one the disclosure recorded"

    def test_every_neutral_game_resolves_to_its_own_stadium(self) -> None:
        measured = _measure()
        still_misrouted = (
            measured["neutral_site_games_total"]
            - measured["neutral_site_games_resolving_to_their_own_stadium"]
        )
        assert still_misrouted == 0, (
            f"{still_misrouted} neutral-site game(s) still resolve elsewhere."
        )

    def test_the_repair_slot_records_the_same_numbers(self) -> None:
        repaired = _recorded_repair()
        measured = _measure()
        assert (
            repaired["neutral_site_games_resolving_to_their_own_stadium"]
            == measured["neutral_site_games_resolving_to_their_own_stadium"]
        )
        assert repaired["still_misrouted"] == 0

    def test_the_repair_supersedes_and_does_not_replace_the_disclosure(self) -> None:
        """T-33.1-19: the record of a measured defect is not overwritten."""
        repaired = _recorded_repair()
        assert repaired["supersedes"] == "HISTORICAL_NEUTRAL_MISRESOLUTION"
        original = phase33_state.HISTORICAL_NEUTRAL_MISRESOLUTION
        assert original["games"] == 91, (
            "the earlier disclosure slot has been edited. The append protocol "
            "forbids it and tests/unit/test_phase33_state_append_once.py enforces "
            "it mechanically; the repair is a NEW slot."
        )
        assert original["distinct_stadium_ids"] == 27
        assert repaired["games_disclosed"] == original["games"]
