"""The permanent ingest-side override for the seven 2025 international games.

Plan 33.2-09 Task 1 (SPEC R8 venue half, T-33.2-09-03, T-33.2-09-07).

WHY AN OVERRIDE AT THE INGEST AND NOT ONLY A STORE REPAIR
--------------------------------------------------------
``transform_schedule_data`` copies the feed's ``stadium_id`` VERBATIM and derives
``venue`` and ``venue_roof`` from the same feed row. The feed records all seven 2025
international games at the US home team's own stadium, so a store repair on its own is
undone by the next ingest of 2025 -- and the Sao Paulo opener goes back to SoFi's
``indoor`` roof, which switches its weather off entirely.

So the ingest reads ``config/international_venue_corrections.toml`` through ONE
resolution function, and the store repair (``scripts/repair_international_venues.py``)
calls the same function. The two cannot disagree.

WHAT IS PINNED HERE
-------------------
* correction: a feed row carrying the home team's stadium comes out corrected, all three
  columns from one decision;
* idempotence: a feed row already carrying the corrected id comes out identical;
* drift: a THIRD id on a recorded game raises ``VenueOverrideDriftError`` OUT of
  ``transform_schedule_data`` -- the broad skip handler must not turn it into a missing
  game;
* no false positive: an unrecorded game passes through unchanged;
* the loader's three refusals.

Rows are built in the captured feed's shape (``tests.fixtures.season_2026``); no network.
ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts import ingest_games
from tests.fixtures import season_2026

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = REPO_ROOT / "config" / "international_venue_corrections.toml"

# The seven games, each with the feed values it arrives with and the triple the ingest
# must write. The expected values are stated here, not read back from the record, so a
# record edited to agree with a wrong ingest still fails.
SEVEN_2025_GAMES: tuple[dict[str, str], ...] = (
    {
        "game_id": "2025_W01_KC@LAC",
        "week": "1",
        "gameday": "2025-09-05",
        "away": "KC",
        "home": "LAC",
        "feed_stadium_id": "LAX01",
        "feed_stadium": "SoFi Stadium",
        "feed_roof": "dome",
        "stadium_id": "SAO00",
        "venue": "Arena Corinthians",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W04_MIN@PIT",
        "week": "4",
        "gameday": "2025-09-28",
        "away": "MIN",
        "home": "PIT",
        "feed_stadium_id": "PIT00",
        "feed_stadium": "Acrisure Stadium",
        "feed_roof": "outdoors",
        "stadium_id": "DUB00",
        "venue": "Croke Park",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W05_MIN@CLE",
        "week": "5",
        "gameday": "2025-10-05",
        "away": "MIN",
        "home": "CLE",
        "feed_stadium_id": "CLE00",
        "feed_stadium": "FirstEnergy Stadium",
        "feed_roof": "outdoors",
        "stadium_id": "LON02",
        "venue": "Tottenham Hotspur Stadium",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W06_DEN@NYJ",
        "week": "6",
        "gameday": "2025-10-12",
        "away": "DEN",
        "home": "NYJ",
        "feed_stadium_id": "NYC01",
        "feed_stadium": "MetLife Stadium",
        "feed_roof": "outdoors",
        "stadium_id": "LON02",
        "venue": "Tottenham Hotspur Stadium",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W07_LA@JAX",
        "week": "7",
        "gameday": "2025-10-19",
        "away": "LA",
        "home": "JAX",
        "feed_stadium_id": "JAX00",
        "feed_stadium": "TIAA Bank Stadium",
        "feed_roof": "outdoors",
        "stadium_id": "LON00",
        "venue": "Wembley Stadium",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W10_ATL@IND",
        "week": "10",
        "gameday": "2025-11-09",
        "away": "ATL",
        "home": "IND",
        "feed_stadium_id": "IND00",
        "feed_stadium": "Lucas Oil Stadium",
        "feed_roof": "closed",
        "stadium_id": "BER00",
        "venue": "Olympiastadion",
        "venue_roof": "outdoor",
    },
    {
        "game_id": "2025_W11_WAS@MIA",
        "week": "11",
        "gameday": "2025-11-16",
        "away": "WAS",
        "home": "MIA",
        "feed_stadium_id": "MIA00",
        "feed_stadium": "Hard Rock Stadium",
        "feed_roof": "outdoors",
        "stadium_id": "MAD01",
        "venue": "Bernabeu",
        "venue_roof": "retractable",
    },
)

THIRD_ID = "ATL97"


def _feed_row(
    *,
    season: int,
    week: int,
    gameday: str,
    away: str,
    home: str,
    location: str,
    stadium_id: str | None,
    stadium: str,
    roof: str,
) -> dict[str, object]:
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "game_type": "REG",
        "week": week,
        "gameday": gameday,
        "gametime": "09:30",
        "away_team": away,
        "home_team": home,
        "location": location,
        "stadium_id": stadium_id,
        "stadium": stadium,
        "roof": roof,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=list(season_2026.CAPTURED_FEED_COLUMNS))


def _seven_feed_rows(*, carry: str) -> pd.DataFrame:
    """The seven games as the feed sends them (`carry="feed"`) or already corrected."""
    rows = []
    for game in SEVEN_2025_GAMES:
        corrected = carry == "corrected"
        rows.append(
            _feed_row(
                season=2025,
                week=int(game["week"]),
                gameday=game["gameday"],
                away=game["away"],
                home=game["home"],
                location="Neutral",
                stadium_id=game["stadium_id"] if corrected else game["feed_stadium_id"],
                stadium=game["venue"] if corrected else game["feed_stadium"],
                roof="outdoors" if corrected else game["feed_roof"],
            )
        )
    return _frame(rows)


def _triples(frame: pd.DataFrame) -> dict[str, tuple[object, object, object]]:
    return {
        row.game_id: (row.stadium_id, row.venue, row.venue_roof)
        for row in frame.itertuples()
    }


def _expected_triples() -> dict[str, tuple[str, str, str]]:
    return {
        game["game_id"]: (game["stadium_id"], game["venue"], game["venue_roof"])
        for game in SEVEN_2025_GAMES
    }


class TestTheCorrectionRecordIsWhatTheIngestReads:
    def test_the_record_holds_one_cited_entry_per_game(self) -> None:
        record = tomllib.loads(RECORD_PATH.read_text(encoding="utf-8"))
        entries = record["correction"]
        assert sorted(e["game_id"] for e in entries) == sorted(_expected_triples())
        for entry in entries:
            assert entry["source_url"].startswith("https://"), entry["game_id"]
            assert entry["source_date"], entry["game_id"]

    def test_the_loader_returns_an_immutable_mapping_keyed_on_game_id(self) -> None:
        overrides = ingest_games.load_venue_overrides()
        assert sorted(overrides) == sorted(_expected_triples())
        with pytest.raises(TypeError):
            overrides["2025_W01_KC@LAC"] = None  # type: ignore[index]

    def test_the_loader_parses_the_record_once_per_process(self) -> None:
        assert (
            ingest_games.load_venue_overrides() is ingest_games.load_venue_overrides()
        )


class TestTheSevenGamesAreCorrectedAtIngest:
    def test_each_feed_row_comes_out_at_the_venue_it_was_played_at(self) -> None:
        ingester = ingest_games.GameDataIngester()
        out = ingester.transform_schedule_data(_seven_feed_rows(carry="feed"))
        assert len(out) == 7
        assert _triples(out) == _expected_triples()

    def test_the_sao_paulo_opener_is_outdoor_not_sofi_indoor(self) -> None:
        ingester = ingest_games.GameDataIngester()
        out = ingester.transform_schedule_data(_seven_feed_rows(carry="feed"))
        opener = out[out["game_id"] == "2025_W01_KC@LAC"].iloc[0]
        assert opener["venue_roof"] == "outdoor", (
            "the opener still resolves SoFi's indoor roof, which switches its weather off"
        )

    def test_the_seven_stay_neutral_site(self) -> None:
        ingester = ingest_games.GameDataIngester()
        out = ingester.transform_schedule_data(_seven_feed_rows(carry="feed"))
        assert out["neutral_site"].all()

    def test_an_already_corrected_feed_row_comes_out_identical(self) -> None:
        """Idempotence: the day the feed fixes its own rows, nothing changes."""
        ingester = ingest_games.GameDataIngester()
        from_feed = ingester.transform_schedule_data(_seven_feed_rows(carry="feed"))
        already = ingester.transform_schedule_data(_seven_feed_rows(carry="corrected"))
        assert _triples(already) == _triples(from_feed) == _expected_triples()


class TestADriftedFeedValueRaisesOutOfTheTransform:
    def test_a_third_id_on_a_recorded_game_propagates(self) -> None:
        frame = _seven_feed_rows(carry="feed")
        frame.loc[1, "stadium_id"] = THIRD_ID
        ingester = ingest_games.GameDataIngester()
        with pytest.raises(ingest_games.VenueOverrideDriftError) as excinfo:
            ingester.transform_schedule_data(frame)
        message = str(excinfo.value)
        assert "2025_W04_MIN@PIT" in message
        for code in (THIRD_ID, "PIT00", "DUB00"):
            assert code in message

    def test_a_missing_feed_id_on_a_recorded_game_propagates(self) -> None:
        frame = _seven_feed_rows(carry="feed")
        frame.loc[0, "stadium_id"] = None
        ingester = ingest_games.GameDataIngester()
        with pytest.raises(ingest_games.VenueOverrideDriftError):
            ingester.transform_schedule_data(frame)

    def test_the_drift_error_is_not_an_identity_error(self) -> None:
        """Its OWN re-raise arm: it must not depend on IdentityColumnError's."""
        assert not issubclass(
            ingest_games.VenueOverrideDriftError, ingest_games.IdentityColumnError
        )


class TestAnUnrecordedGamePassesThroughUnchanged:
    @pytest.mark.parametrize(
        "row",
        [
            # A 2024 international game the feed already records correctly.
            _feed_row(
                season=2024,
                week=5,
                gameday="2024-10-06",
                away="NYJ",
                home="MIN",
                location="Neutral",
                stadium_id="LON02",
                stadium="Tottenham Stadium",
                roof="outdoors",
            ),
            # A 2026 international game, where the feed says dome on an open-air venue.
            _feed_row(
                season=2026,
                week=9,
                gameday="2026-11-05",
                away="DET",
                home="LA",
                location="Neutral",
                stadium_id="MEL00",
                stadium="Melbourne Cricket Ground",
                roof="dome",
            ),
            # An ordinary home game.
            _feed_row(
                season=2025,
                week=3,
                gameday="2025-09-21",
                away="CIN",
                home="MIN",
                location="Home",
                stadium_id="MIN01",
                stadium="U.S. Bank Stadium",
                roof="dome",
            ),
        ],
        ids=["2024_london", "2026_melbourne", "2025_home_game"],
    )
    def test_all_three_columns_are_the_feed_derived_values(
        self, row: dict[str, object]
    ) -> None:
        ingester = ingest_games.GameDataIngester()
        out = ingester.transform_schedule_data(_frame([row])).iloc[0]
        assert out["stadium_id"] == row["stadium_id"]
        assert out["venue"] == row["stadium"]
        assert out["venue_roof"] == ingester._get_venue_roof_type(
            str(row["stadium"]), str(row["roof"]), stadium_id=str(row["stadium_id"])
        )

    def test_the_resolver_returns_feed_values_for_an_unrecorded_game(self) -> None:
        ingester = ingest_games.GameDataIngester()
        triple = ingest_games.resolve_venue_override(
            "2025_W03_CIN@MIN",
            "MIN01",
            "U.S. Bank Stadium",
            "dome",
            roof_resolver=ingester._get_venue_roof_type,
        )
        assert triple == ("MIN01", "U.S. Bank Stadium", "indoor")


def _write_record(tmp_path: Path, entries: list[dict[str, str]]) -> Path:
    lines: list[str] = []
    for entry in entries:
        lines.append("[[correction]]")
        lines.extend(f'{key} = "{value}"' for key, value in entry.items())
        lines.append("")
    path = tmp_path / "corrections.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _good_entry(**overrides: str) -> dict[str, str]:
    entry = {
        "game_id": "2025_W04_MIN@PIT",
        "old_stadium_id": "PIT00",
        "new_stadium_id": "DUB00",
        "venue_name": "Croke Park",
        "city": "Dublin",
        "country": "Ireland",
        "source_url": "https://example.org/croke-park",
        "source_date": "2025-02-07",
    }
    entry.update(overrides)
    return entry


class TestTheLoaderRefusesByName:
    def test_a_well_formed_planted_record_loads(self, tmp_path: Path) -> None:
        path = _write_record(tmp_path, [_good_entry()])
        overrides = ingest_games.load_venue_overrides(path)
        assert overrides["2025_W04_MIN@PIT"].new_stadium_id == "DUB00"

    def test_an_empty_source_url_is_refused(self, tmp_path: Path) -> None:
        path = _write_record(tmp_path, [_good_entry(source_url="")])
        with pytest.raises(ingest_games.VenueOverrideRecordError, match="source_url"):
            ingest_games.load_venue_overrides(path)

    def test_a_duplicate_game_id_is_refused(self, tmp_path: Path) -> None:
        path = _write_record(tmp_path, [_good_entry(), _good_entry()])
        with pytest.raises(
            ingest_games.VenueOverrideRecordError, match="2025_W04_MIN@PIT"
        ):
            ingest_games.load_venue_overrides(path)

    def test_a_new_stadium_id_absent_from_venues_json_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = _write_record(tmp_path, [_good_entry(new_stadium_id="XXX99")])
        with pytest.raises(ingest_games.VenueOverrideRecordError, match="XXX99"):
            ingest_games.load_venue_overrides(path)

    def test_a_venue_name_that_disagrees_with_venues_json_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Two answers for one venue name: the record must agree with the file."""
        path = _write_record(tmp_path, [_good_entry(venue_name="Croker")])
        with pytest.raises(ingest_games.VenueOverrideRecordError, match="Croker"):
            ingest_games.load_venue_overrides(path)
