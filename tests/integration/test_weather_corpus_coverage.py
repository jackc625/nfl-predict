"""The promoted silver weather corpus is COMPLETE, PROVENANCED and COVERAGE-FLAGGED.

WHAT THIS MODULE IS FOR (SPEC R2, R4)
-------------------------------------
Plan 33.1-06 fetched 4,847 ERA5 observations and promoted 6,499 rows into
``data/silver/weather.parquet`` in a single bracketed upsert. This module is the
standing assertion that what landed there is what was supposed to land there:

* the corpus covers the PINNED FEED EXACTLY -- no extra game, no absent game, and
  a failure names the symmetric difference rather than counting it;
* every row carries a ``weather_source`` from the CLOSED vocabulary, so no row's
  provenance is unknown;
* every row carries a non-null ``weather_coverage``, so the three states R4 keeps
  apart stay apart.

THE THREE STATES, AND WHY THE POPULATIONS ARE DERIVED FROM THE FEED
-------------------------------------------------------------------
``tests.phase33_state.WEATHER_NULL_STATE_MATRIX`` records Ruling J's three states,
keyed on the ``(weather_coverage, is_outdoor)`` pair:

    covered_indoor            coverage True,  is_outdoor False -- a dome or a
                              `closed` roof. Weather does not apply, so there is
                              nothing to measure and ``temp_f`` is NULL.
    covered_outdoor_observed  coverage True,  is_outdoor True  -- a reading.
    uncovered_outdoor_absent  coverage False, is_outdoor True  -- the archive had
                              no observation. Every column NULL, and never a number.

Before the coverage flag existed all three collapsed into one, because a missing
record was written with ``is_outdoor=False`` and ``temp_f=65.0``.

THE INDOOR POPULATION IS READ FROM THE PINNED FEED'S OWN ``roof`` VALUES, not from
the frame's ``is_outdoor`` column. Asserting that the frame agrees with itself
proves nothing; the claim R4 makes is that the frame agrees with the FEED, and the
feed is the independent instrument. That is also what catches a regression of the
per-game roof rule: five stadiums record ``closed`` on some games and ``open`` on
others, and a venue-level rule would put all of them on one side.

THE CORPUS IS NOW THE ARCHIVED DAY-BEFORE FORECAST (Plan 33.2-12, p332_ rung 4). Rung 4
replaced every 2002-2025 ERA5 observation with the 12 UTC NWS MOS bulletin of the day before
kickoff (``scripts/weather_from_mos.py``), so each row now reads ``historical_forecast`` and a
covered outdoor row is a FORECAST, not a reading. The three states and the feed-derived indoor
population are unchanged, with two refinements recorded where they bite:

* THE CORPUS IS THE 2002-2025 SLICE. The table also holds the LIVE path's 2026 rows (Open-Meteo,
  ``weather_source = "forecast"``), which were never part of this corpus; before rung 4 they
  already made the "only these games" and "only archive" checks read red. The corpus fixture is
  the history slice, which is what these assertions were always about.
* THE ROOF IS READ AT THE GAME'S VENUE. For the seven 2025 games played abroad, the feed's row
  still names the US stadium (Plan 33.2-09 corrected the venue in silver ``games``), so its
  ``roof`` describes a stadium the game was not played in -- it put Sao Paulo under SoFi's dome
  and Berlin under Lucas Oil's closed roof. A game whose feed stadium differs from its silver
  venue takes that venue's own roof type instead.

READ ONLY. Nothing here writes anything. The module requests
``data_boundary_guard`` so that is a proven property rather than an intention.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.backfill_historical_weather import (
    CORPUS_FIRST_SEASON,
    CORPUS_LAST_SEASON,
    load_pinned_game_facts,
)
from scripts.ingest_weather import WEATHER_SOURCE_VOCABULARY

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER_WEATHER = REPO_ROOT / "data" / "silver" / "weather.parquet"
SILVER_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"
VENUES_JSON = REPO_ROOT / "data" / "venues.json"

# The whole corpus, and the 2025 slice the promotion's acceptance names explicitly.
EXPECTED_CORPUS_ROWS = 6499
EXPECTED_2025_ROWS = 285

# The roof values that mean weather DOES NOT APPLY, spelled as the nflverse feed
# spells them. `outdoors` and `open` are the complement; there is no fifth value.
INDOOR_ROOFS = frozenset({"dome", "closed"})

# The skip says WHICH COMMAND produces the file, so a skipped run is actionable
# rather than silent -- and it carries the word "absent", which
# `tests/conftest.is_evidence_backed_skip` reads, so a checkout without the data
# is TOLD that this control did not run rather than being shown a green suite.
_SKIP_REASON = (
    f"the promoted silver weather corpus is absent at "
    f"{SILVER_WEATHER.relative_to(REPO_ROOT).as_posix()} -- data/ is gitignored, so "
    "a fresh checkout does not have it. Produce it with:\n"
    "    uv run python -m scripts.backfill_historical_weather --bracket run\n"
    "    uv run python -m scripts.backfill_historical_weather --bracket promote"
)

pytestmark = pytest.mark.skipif(not SILVER_WEATHER.is_file(), reason=_SKIP_REASON)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    """The 2002-2025 history slice of silver weather, read once for the whole module."""
    frame = pd.read_parquet(SILVER_WEATHER, engine="pyarrow")
    seasons = frame["game_id"].str.slice(0, 4).astype(int)
    return frame[seasons.between(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON)]


@pytest.fixture(scope="module")
def feed() -> pd.DataFrame:
    """The pinned schedules for the whole corpus, indexed by ``game_id``.

    The INDEPENDENT instrument. It carries each game's own ``roof``, which is what
    decides whether weather applies -- and it is pinned, so this fixture describes
    the same corpus a year from now.
    """
    seasons = list(range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1))
    return load_pinned_game_facts(seasons).set_index("game_id")


@pytest.fixture(scope="module")
def indoor_ids(feed) -> set[str]:
    """Games weather does not apply to: a dome or closed roof, read at the game's venue.

    The feed's own ``roof`` decides, except where the feed's row names a different stadium
    from the game's silver venue (the corrected 2025 games abroad): there the venue's own
    ``roof_type`` in ``data/venues.json`` decides.
    """
    games = pd.read_parquet(SILVER_GAMES, engine="pyarrow").set_index("game_id")
    venues = json.loads(VENUES_JSON.read_text(encoding="utf-8"))["venues"]
    roof_type = {str(v["stadium_id"]): str(v["roof_type"]) for v in venues}
    indoor: set[str] = set()
    for game_id, fact in feed.iterrows():
        venue = str(games.loc[game_id, "stadium_id"])
        if str(fact["stadium_id"]) == venue:
            if str(fact["roof"]).lower().strip() in INDOOR_ROOFS:
                indoor.add(str(game_id))
        elif roof_type[venue] == "indoor":
            indoor.add(str(game_id))
    return indoor


class TestTheCorpusCoversThePinnedFeedExactly:
    """SPEC R2: 6,499 rows over 2002-2025, and the SAME 6,499 games the feed names."""

    def test_the_corpus_holds_every_game_and_only_those_games(
        self, corpus, feed, data_boundary_guard
    ):
        promoted = set(corpus["game_id"])
        pinned = set(feed.index)

        missing = sorted(pinned - promoted)
        extra = sorted(promoted - pinned)

        # NAMED, not counted. "37 games missing" sends the reader back to the data
        # to find out which; a set difference already knows.
        assert not missing, (
            f"{len(missing)} pinned game(s) have NO weather row. A partial corpus "
            f"promoted as if complete is threat T-33.1-38. First ten: {missing[:10]}"
        )
        assert not extra, (
            f"{len(extra)} weather row(s) name a game the pinned feed does not "
            f"have. First ten: {extra[:10]}"
        )
        assert len(corpus) == EXPECTED_CORPUS_ROWS, (
            f"the corpus holds {len(corpus)} rows, not {EXPECTED_CORPUS_ROWS}. The "
            "game-id sets match, so this is a DUPLICATE row rather than a coverage "
            "gap -- the promotion deduplicates on game_id, keeping the last."
        )

    def test_it_spans_2002_through_2025_with_the_whole_of_2025(
        self, corpus, data_boundary_guard
    ):
        seasons = corpus["game_id"].str.slice(0, 4).astype(int)

        assert int(seasons.min()) == CORPUS_FIRST_SEASON
        assert int(seasons.max()) == CORPUS_LAST_SEASON
        assert set(seasons) == set(
            range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1)
        ), (
            "a season inside the corpus range has NO rows: "
            f"{sorted(set(range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1)) - set(seasons))}"
        )
        assert int((seasons == 2025).sum()) == EXPECTED_2025_ROWS, (
            "2025 is the season the whole phase exists to stop hiding; it must "
            f"carry all {EXPECTED_2025_ROWS} games, not "
            f"{int((seasons == 2025).sum())}."
        )


class TestEveryRowIsProvenanced:
    """SPEC R2: a closed `weather_source` vocabulary, and nothing outside it."""

    def test_every_weather_source_is_in_the_closed_vocabulary(
        self, corpus, data_boundary_guard
    ):
        observed = set(corpus["weather_source"].dropna().unique())
        outside = sorted(observed - set(WEATHER_SOURCE_VOCABULARY))

        assert not outside, (
            f"weather_source values outside the closed vocabulary: {outside}. The "
            f"allowed set is {list(WEATHER_SOURCE_VOCABULARY)}. A value outside it "
            "means a row whose provenance this repository cannot name."
        )

    def test_no_row_has_a_null_weather_source(self, corpus, data_boundary_guard):
        nulls = corpus[corpus["weather_source"].isna()]
        assert len(nulls) == 0, (
            f"{len(nulls)} row(s) carry a NULL weather_source. First ten game ids: "
            f"{sorted(nulls['game_id'])[:10]}"
        )

    def test_the_whole_corpus_came_from_the_archive(self, corpus, data_boundary_guard):
        """Every 2002-2025 row is the ARCHIVED DAY-BEFORE FORECAST (since rung 4).

        Not a tautology: `archive` and `forecast` are both in the vocabulary and both
        reachable through other code paths. An `archive` row here would be an ERA5
        observation surviving under a forecast's name; a `forecast` row would be a
        live-path row mixed into the historical corpus. (The test keeps its name: the
        corpus still comes from an archive -- IEM's archive of the bulletins.)
        """
        assert set(corpus["weather_source"].unique()) == {"historical_forecast"}, (
            "the promoted historical corpus carries a non-archive provenance: "
            f"{sorted(set(corpus['weather_source'].unique()))}"
        )


class TestTheThreeCoverageStatesStayApart:
    """SPEC R4: a dome, an absent observation and a reading are DIFFERENT facts."""

    def test_no_row_has_a_null_coverage_flag(self, corpus, data_boundary_guard):
        nulls = corpus[corpus["weather_coverage"].isna()]
        assert len(nulls) == 0, (
            f"{len(nulls)} row(s) carry a NULL weather_coverage. The flag is what "
            "keeps 'weather does not apply' apart from 'the archive had no "
            "reading'; a null collapses them again."
        )

    def test_every_indoor_game_is_covered_and_carries_no_temperature(
        self, corpus, indoor_ids, data_boundary_guard
    ):

        indoor = corpus[corpus["game_id"].isin(indoor_ids)]
        assert len(indoor) == len(indoor_ids), (
            f"{len(indoor_ids)} games read dome/closed in the feed but only "
            f"{len(indoor)} of them have a weather row."
        )

        uncovered = sorted(
            indoor.loc[~indoor["weather_coverage"].astype(bool), "game_id"]
        )
        assert not uncovered, (
            "a dome or closed-roof game reads weather_coverage FALSE. Coverage "
            "TRUE is correct for an indoor game (Ruling E): the record is "
            "COMPLETE -- weather does not apply -- which is a different fact from "
            f"an absent observation. First ten: {uncovered[:10]}"
        )

        numeric = sorted(indoor.loc[indoor["temp_f"].notna(), "game_id"])
        assert not numeric, (
            "a dome or closed-roof game carries a NUMERIC temp_f. There is no "
            "observation for a game played indoors, and a number here is the "
            f"65.0-default defect wearing a different value. First ten: {numeric[:10]}"
        )

    def test_every_observed_outdoor_game_is_covered_and_carries_a_temperature(
        self, corpus, feed, indoor_ids, data_boundary_guard
    ):
        outdoor_ids = set(feed.index) - indoor_ids

        outdoor = corpus[corpus["game_id"].isin(outdoor_ids)]
        observed = outdoor[outdoor["weather_coverage"].astype(bool)]

        missing_temp = sorted(observed.loc[observed["temp_f"].isna(), "game_id"])
        assert not missing_temp, (
            "an outdoor game reads weather_coverage TRUE but carries no temp_f. "
            "Those two statements cannot both be right: coverage TRUE on an "
            "outdoor row means an observation was recorded. First ten: "
            f"{missing_temp[:10]}"
        )

    def test_an_absent_observation_is_outdoor_covered_false_and_entirely_null(
        self, corpus, indoor_ids, data_boundary_guard
    ):
        absent = corpus[~corpus["weather_coverage"].astype(bool)]
        if absent.empty:
            pytest.skip(
                "the corpus holds no absent observation, so there is no row in "
                "the uncovered_outdoor_absent state to check. This is a complete "
                "result, not a gap: the archive answered for every outdoor game."
            )

        wrongly_indoor = sorted(set(absent["game_id"]) & indoor_ids)
        assert not wrongly_indoor, (
            "an INDOOR game reads weather_coverage FALSE. An absent observation "
            "is only reachable for a game that was actually fetched, and an "
            f"indoor game is never fetched. First ten: {wrongly_indoor[:10]}"
        )

        numeric = sorted(absent.loc[absent["temp_f"].notna(), "game_id"])
        assert not numeric, (
            "an absent observation carries a NUMERIC temp_f. Every column of an "
            "absent row is NULL; a number would be a stand-in under a different "
            f"name, which SPEC prohibition 1 forbids. First ten: {numeric[:10]}"
        )

        assert absent["is_outdoor"].astype(bool).all(), (
            "an absent observation reads is_outdoor FALSE, which is exactly the "
            "conflation R4 exists to end: before this phase a missing record was "
            "written with is_outdoor False and was therefore indistinguishable "
            "from a dome."
        )

    def test_the_three_states_partition_the_corpus(
        self, corpus, indoor_ids, data_boundary_guard
    ):
        """The three populations are disjoint and sum to the whole corpus.

        The arithmetic is the point: if they did not sum, some row would be in a
        FOURTH state nobody named -- which is how the 65.0 default survived for
        as long as it did.
        """

        covered = corpus["weather_coverage"].astype(bool)
        is_indoor = corpus["game_id"].isin(indoor_ids)

        covered_indoor = int((covered & is_indoor).sum())
        covered_outdoor_observed = int((covered & ~is_indoor).sum())
        uncovered_outdoor_absent = int((~covered).sum())

        total = covered_indoor + covered_outdoor_observed + uncovered_outdoor_absent
        assert total == EXPECTED_CORPUS_ROWS, (
            "the three coverage states do not partition the corpus: "
            f"{covered_indoor} indoor + {covered_outdoor_observed} observed + "
            f"{uncovered_outdoor_absent} absent = {total}, not "
            f"{EXPECTED_CORPUS_ROWS}."
        )
        assert covered_indoor > 0 and covered_outdoor_observed > 0, (
            "one of the two covered states is EMPTY, so this assertion is not "
            f"exercising the partition: indoor={covered_indoor}, "
            f"observed={covered_outdoor_observed}"
        )
