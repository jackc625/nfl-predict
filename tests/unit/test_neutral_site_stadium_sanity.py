"""STANDING GUARD: no neutral-site game may sit at the home team's usual stadium.

Plan 33.2-09 Task 2 (SPEC R8 venue half, T-33.2-09-01, T-33.2-09-08).

THE DEFECT THIS GUARDS
----------------------
All seven 2025 games played outside the United States were stored at the US home team's
own stadium with ``neutral_site=True``. The shape is mechanical: a neutral-site game whose
``stadium_id`` equals the home team's MODAL stadium over its non-neutral home games that
season. 2024 had 5 non-US venue rows and 2026 has 8; 2025 had 0, and nothing noticed.

A LEGITIMATE CASE EXISTS, SO THE RULE HAS A TYPED, CITED REGISTRY
-----------------------------------------------------------------
Super Bowl LV (``2020_W21_KC@TB``) was played at TAM00, Tampa Bay's own stadium, and the
feed designates TB the home team. It is recorded in
``config/neutral_site_venue_exceptions.toml`` with a ``reason_type`` from a CLOSED
vocabulary and a source. An entry whose game is no longer a neutral-at-usual case fails as
stale, so the registry cannot grow into an allow-list.

THE SEVEN CORRECTED GAMES ARE ASSERTED, NOT EXEMPTED
----------------------------------------------------
Each must carry exactly what ``scripts.ingest_games.resolve_venue_override`` returns for
it given the feed's wrong value -- which is what a re-ingest of 2025 would write. A store
and an ingest that disagree fail here, and a re-ingest that restored a wrong venue fails
the neutral-at-usual scan rather than being waved through by the record meant to stop it.

READ-ONLY. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts import ingest_games

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"
VENUES_PATH = REPO_ROOT / "data" / "venues.json"
EXCEPTIONS_PATH = REPO_ROOT / "config" / "neutral_site_venue_exceptions.toml"

FIRST_SEASON = 2002
LAST_SEASON = 2026

# The CLOSED reason vocabulary. There is deliberately no reason for a corrected
# international game: those are asserted positively below, never exempted.
EXCEPTION_REASON_TYPES: frozenset[str] = frozenset(
    {"super_bowl_home_designation", "verified_neutral_at_home"}
)

# Measured 2026-09-21: the one legitimate case after the correction.
SUPER_BOWL_LV = "2020_W21_KC@TB"

US_COUNTRY_VALUES: frozenset[str] = frozenset({"US", "USA", "UNITED STATES"})


def _silver_games() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        pytest.skip("silver games is not built on this checkout")
    games = pd.read_parquet(GAMES_PATH)
    return games[games["season"].between(FIRST_SEASON, LAST_SEASON)]


def _exceptions() -> list[dict[str, str]]:
    return tomllib.loads(EXCEPTIONS_PATH.read_text(encoding="utf-8"))["exception"]


def neutral_games_at_usual_stadium(games: pd.DataFrame) -> list[str]:
    """Neutral-site games whose stadium is the home team's modal non-neutral stadium."""
    neutral = games["neutral_site"].astype(bool)
    usual = (
        games[~neutral]
        .groupby(["season", "home_team"])["stadium_id"]
        .agg(lambda ids: ids.mode().iat[0])
        .rename("usual_stadium_id")
        .reset_index()
    )
    merged = games[neutral].merge(usual, on=["season", "home_team"], how="left")
    return sorted(
        merged.loc[merged["stadium_id"] == merged["usual_stadium_id"], "game_id"]
    )


def violations(games: pd.DataFrame, exceptions: list[dict[str, str]]) -> list[str]:
    """Neutral-at-usual games NOT cleared by a well-formed registry entry."""
    cleared = {
        entry["game_id"]
        for entry in exceptions
        if entry.get("reason_type") in EXCEPTION_REASON_TYPES
        and entry.get("source_url")
    }
    return [
        game for game in neutral_games_at_usual_stadium(games) if game not in cleared
    ]


class TestTheRegistryIsTypedAndCited:
    def test_every_entry_carries_a_pinned_reason_type(self) -> None:
        bad = [
            e["game_id"]
            for e in _exceptions()
            if e.get("reason_type") not in EXCEPTION_REASON_TYPES
        ]
        assert not bad, f"entries {bad!r} carry a reason_type outside the vocabulary"

    def test_every_entry_carries_a_source(self) -> None:
        bad = [e["game_id"] for e in _exceptions() if not e.get("source_url")]
        assert not bad, f"entries {bad!r} carry no source_url"

    def test_super_bowl_lv_is_recorded_as_a_super_bowl_home_designation(self) -> None:
        entries = {e["game_id"]: e for e in _exceptions()}
        assert entries[SUPER_BOWL_LV]["reason_type"] == "super_bowl_home_designation"
        assert entries[SUPER_BOWL_LV]["stadium_id"] == "TAM00"

    def test_no_entry_is_a_corrected_international_game(self) -> None:
        corrected = set(ingest_games.load_venue_overrides())
        exempted = {e["game_id"] for e in _exceptions()}
        assert not corrected & exempted, (
            "a corrected international game is exempted; it must be asserted instead"
        )

    def test_each_entry_names_the_stadium_silver_stores(self) -> None:
        games = _silver_games().set_index("game_id")
        for entry in _exceptions():
            assert games.loc[entry["game_id"], "stadium_id"] == entry["stadium_id"]


class TestNoNeutralSiteGameSitsAtTheHomeTeamsUsualStadium:
    def test_there_are_neutral_site_games_to_check(self) -> None:
        """Non-vacuity: 99 neutral-site games at measurement."""
        assert int(_silver_games()["neutral_site"].astype(bool).sum()) > 0

    def test_every_neutral_at_usual_game_has_a_registry_entry(self) -> None:
        unexcepted = violations(_silver_games(), _exceptions())
        assert not unexcepted, (
            f"neutral-site game(s) {unexcepted!r} sit at the home team's usual stadium "
            "with no typed, cited entry in config/neutral_site_venue_exceptions.toml. "
            "This is the shape that stored every 2025 international game at a US "
            "stadium; correct the venue, or record why it is legitimate."
        )

    def test_no_registry_entry_is_stale(self) -> None:
        current = set(neutral_games_at_usual_stadium(_silver_games()))
        stale = sorted({e["game_id"] for e in _exceptions()} - current)
        assert not stale, (
            f"registry entries {stale!r} are no longer neutral-at-usual cases; remove "
            "them rather than let the registry grow into an allow-list"
        )

    def test_no_2025_game_is_left_at_the_home_teams_stadium(self) -> None:
        left = [
            g
            for g in neutral_games_at_usual_stadium(_silver_games())
            if g.startswith("2025_")
        ]
        assert not left, f"2025 neutral-site game(s) {left!r} sit at the home stadium"


class TestTheSevenCorrectedGamesAreAssertedPositively:
    def test_silver_equals_what_a_re_ingest_of_2025_would_write(self) -> None:
        games = _silver_games().set_index("game_id")
        overrides = ingest_games.load_venue_overrides()
        roof_resolver = ingest_games.GameDataIngester()._get_venue_roof_type
        assert len(overrides) == 7
        for game_id, override in overrides.items():
            expected = ingest_games.resolve_venue_override(
                game_id,
                override.old_stadium_id,
                None,
                None,
                roof_resolver=roof_resolver,
                overrides=overrides,
            )
            stored = tuple(games.loc[game_id, ["stadium_id", "venue", "venue_roof"]])
            assert stored == expected, (
                f"{game_id}: silver {stored!r}, ingest {expected!r}"
            )

    def test_a_real_re_ingest_of_the_pinned_2025_feed_reproduces_silver(self) -> None:
        """The whole 2025 season through the REAL transform, against silver.

        Measured 2026-09-21 before the repair: exactly the seven recorded games differed
        (285 games). After it, every 2025 game's three venue columns must agree, which is
        what "the next ingest cannot restore the defect" means in practice.
        """
        from data import upstream_pin

        try:
            feed = upstream_pin.load_schedules([2025])
        except upstream_pin.UpstreamPinError as exc:
            pytest.skip(f"the pinned 2025 schedule is unavailable: {exc}")
        ingested = ingest_games.GameDataIngester().transform_schedule_data(feed)
        columns = ["stadium_id", "venue", "venue_roof"]
        ingested = ingested.set_index("game_id")[columns].sort_index()
        games = _silver_games()
        stored = games[games["season"] == 2025].set_index("game_id")[columns]
        stored = stored.sort_index()
        assert list(ingested.index) == list(stored.index)
        disagree = ingested.index[(ingested != stored).any(axis=1)].tolist()
        assert not disagree, f"a re-ingest of 2025 would rewrite {disagree!r}"

    def test_the_seven_stay_neutral_site(self) -> None:
        games = _silver_games().set_index("game_id")
        for game_id in ingest_games.load_venue_overrides():
            assert bool(games.loc[game_id, "neutral_site"]) is True

    def test_the_sao_paulo_opener_is_outdoor(self) -> None:
        games = _silver_games().set_index("game_id")
        assert games.loc["2025_W01_KC@LAC", "venue_roof"] == "outdoor"


class TestTheGuardIsNotVacuous:
    def test_a_planted_neutral_game_at_the_usual_stadium_is_flagged(self) -> None:
        games = _silver_games().copy()
        override = ingest_games.load_venue_overrides()["2025_W04_MIN@PIT"]
        games.loc[games["game_id"] == override.game_id, "stadium_id"] = (
            override.old_stadium_id
        )
        assert override.game_id in violations(games, _exceptions())

    def test_a_corrected_game_is_not_flagged(self) -> None:
        assert "2025_W04_MIN@PIT" not in violations(_silver_games(), _exceptions())

    def test_super_bowl_lv_is_cleared_by_its_entry_and_only_by_it(self) -> None:
        games = _silver_games()
        assert SUPER_BOWL_LV in neutral_games_at_usual_stadium(games)
        assert SUPER_BOWL_LV not in violations(games, _exceptions())
        without = [e for e in _exceptions() if e["game_id"] != SUPER_BOWL_LV]
        assert SUPER_BOWL_LV in violations(games, without), (
            "Super Bowl LV is not flagged even without its entry, so the rule never "
            "looked at it"
        )


class TestThe2025SeasonHasNonUsVenueRows:
    def test_2025_carries_non_us_venue_rows(self) -> None:
        """The regression that produced this defect: 2024 had 5, 2026 has 8, 2025 had 0."""
        venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
        non_us = {
            v["stadium_id"]
            for v in venues
            if str(v["country"]).upper() not in US_COUNTRY_VALUES
        }
        games = _silver_games()
        count = int(games.loc[games["season"] == 2025, "stadium_id"].isin(non_us).sum())
        assert count == 7, f"2025 carries {count} non-US venue rows, expected 7"
