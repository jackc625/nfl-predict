"""Real `stadium_id`, `neutral_site` and `season_type` at ingest.

Phase 33, Plan 33-06 Task 3 (COLD-09, D33-15/D33-17, T-33-27).

THE DEFECT
----------
``scripts/ingest_games.transform_schedule_data`` read
``row.get("season_type", "Regular")`` and ``row.get("neutral_site", False)``. The
nflverse schedule frame carries NEITHER key -- it carries ``location`` and
``game_type`` -- so the ``.get`` default fired on every row and all 6,499 silver
rows read ``season_type == 'Regular'`` and ``neutral_site == False``, including every
WC/DIV/CON/SB game ever played and all eight of 2026's international games. A default
that fires 100% of the time is not a default; it is a constant nobody chose.

``stadium_id`` was not carried at all, which is why the venue routing this plan adds
had nothing to route on downstream.

THE DERIVATIONS
---------------
* ``neutral_site`` is ``location == 'Neutral'``. An ABSENT ``location`` RAISES rather
  than defaulting -- an absent column is exactly how the present defect arose, and
  defaulting again would rebuild it one layer up.
* ``season_type`` is a TOTAL map over the five ``game_type`` values, raising by name
  on an unknown sixth. It is a COARSENING (two values against five), not a duplicate:
  the season-close readout partitions regular-versus-post, and
  ``utils/similar_games.py`` already reads ``season_type`` with a ``game_type``
  fallback and today receives a constant.
* ``stadium_id`` passes through UNCHANGED -- no normalization, no casefolding --
  because R11's matching is exact and case-sensitive.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import ingest_games
from tests import phase33_state
from tests.fixtures import season_2026

# The five game_type values the feed actually uses, and the two-value partition they
# coarsen to. Stated here so the totality assertion has something to be checked
# against rather than only the implementation's own dictionary.
REGULAR_GAME_TYPES = ("REG",)
POSTSEASON_GAME_TYPES = ("WC", "DIV", "CON", "SB")


def _require_capture() -> None:
    try:
        season_2026.load_captured_schedule()
    except season_2026.CapturedScheduleUnavailableError as exc:
        pytest.skip(str(exc))


class TestSeasonTypeDerivation:
    """D33-17: derived from `game_type`, and the two can never disagree."""

    @pytest.mark.parametrize("game_type", REGULAR_GAME_TYPES)
    def test_regular_season_types(self, game_type: str) -> None:
        assert ingest_games._derive_season_type({"game_type": game_type}) == "Regular"

    @pytest.mark.parametrize("game_type", POSTSEASON_GAME_TYPES)
    def test_postseason_types(self, game_type: str) -> None:
        assert (
            ingest_games._derive_season_type({"game_type": game_type}) == "Postseason"
        )

    def test_the_mapping_is_total_over_the_five_known_values(self) -> None:
        """Totality is the property; a partial map would default silently again."""
        mapped = {
            game_type: ingest_games._derive_season_type({"game_type": game_type})
            for game_type in (*REGULAR_GAME_TYPES, *POSTSEASON_GAME_TYPES)
        }
        assert set(mapped.values()) == {"Regular", "Postseason"}
        assert sorted(k for k, v in mapped.items() if v == "Regular") == ["REG"]
        assert sorted(k for k, v in mapped.items() if v == "Postseason") == [
            "CON",
            "DIV",
            "SB",
            "WC",
        ]

    def test_an_unknown_sixth_value_raises_and_names_it(self) -> None:
        with pytest.raises(ingest_games.IdentityColumnError, match="XX"):
            ingest_games._derive_season_type({"game_type": "XX"})

    def test_an_absent_game_type_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ingest_games.IdentityColumnError, match="game_type"):
            ingest_games._derive_season_type({})

    def test_season_type_and_game_type_can_never_disagree(self) -> None:
        """The partition is a FUNCTION of game_type, so disagreement is unreachable.

        Asserted by driving every known value through the derivation twice and
        checking the answer is stable and partition-consistent -- the mechanical
        form of "these two columns can never contradict each other".
        """
        for game_type in (*REGULAR_GAME_TYPES, *POSTSEASON_GAME_TYPES):
            first = ingest_games._derive_season_type({"game_type": game_type})
            second = ingest_games._derive_season_type({"game_type": game_type})
            assert first == second
            is_regular = game_type in REGULAR_GAME_TYPES
            assert (first == "Regular") is is_regular


class TestNeutralSiteDerivation:
    """It is `location == 'Neutral'`, and an absent `location` is a refusal."""

    def test_neutral_is_true(self) -> None:
        assert ingest_games._derive_neutral_site({"location": "Neutral"}) is True

    def test_home_is_false(self) -> None:
        assert ingest_games._derive_neutral_site({"location": "Home"}) is False

    def test_an_absent_location_raises_rather_than_defaulting(self) -> None:
        """An absent column is EXACTLY how the present defect arose."""
        with pytest.raises(ingest_games.IdentityColumnError, match="location"):
            ingest_games._derive_neutral_site({})

    def test_a_null_location_raises(self) -> None:
        with pytest.raises(ingest_games.IdentityColumnError, match="location"):
            ingest_games._derive_neutral_site({"location": None})

    def test_the_match_is_exact(self) -> None:
        """'neutral' lowercase is not the feed's value and must not be treated as it."""
        assert ingest_games._derive_neutral_site({"location": "neutral"}) is False


class TestTheCaptured2026Schedule:
    """The derivations against the real feed, not against a constructed row."""

    def test_exactly_eight_rows_are_neutral(self) -> None:
        _require_capture()
        transformed = season_2026.transform_captured_schedule()
        assert (
            int(transformed["neutral_site"].sum())
            == phase33_state.NEUTRAL_SITE_GAME_COUNT_2026
        ), (
            f"{int(transformed['neutral_site'].sum())} rows read neutral_site True, "
            f"expected {phase33_state.NEUTRAL_SITE_GAME_COUNT_2026}. Before this plan "
            "the answer was 0 for all 272 rows."
        )

    def test_the_eight_are_the_recorded_international_games(self) -> None:
        _require_capture()
        raw = season_2026.load_captured_schedule()
        transformed = season_2026.transform_captured_schedule()
        codes = [
            code
            for code, neutral in zip(
                raw["stadium_id"], transformed["neutral_site"], strict=True
            )
            if neutral
        ]
        assert codes == list(phase33_state.INTERNATIONAL_STADIUM_IDS)

    def test_all_272_captured_rows_are_regular_season(self) -> None:
        _require_capture()
        transformed = season_2026.transform_captured_schedule()
        assert set(transformed["season_type"]) == {"Regular"}, (
            "the capture is 272 REG rows over weeks 1-18; weeks 19-22 are unseeded "
            "at capture time. A non-Regular value here means the capture moved."
        )

    def test_stadium_id_reaches_the_transformed_frame_unchanged(self) -> None:
        _require_capture()
        raw = season_2026.load_captured_schedule()
        transformed = season_2026.transform_captured_schedule()
        assert "stadium_id" in transformed.columns
        assert list(transformed["stadium_id"]) == list(raw["stadium_id"]), (
            "stadium_id was altered in transit. R11's matching is exact and "
            "case-sensitive, so any normalization here breaks the routing that "
            "reads it."
        )

    def test_the_frame_carries_all_three_identity_columns(self) -> None:
        _require_capture()
        transformed = season_2026.transform_captured_schedule()
        for column in phase33_state.PLAN_33_06_IDENTITY_COLUMNS:
            assert column in transformed.columns, column


class TestTheConstructedPostseasonFixture:
    """Postseason has to be BUILT: the capture holds no week-19-and-later row."""

    def test_the_wild_card_rows_map_to_postseason(self) -> None:
        ingester = ingest_games.GameDataIngester()
        transformed = ingester.transform_schedule_data(
            season_2026.WEEK_19_POSTSEASON_FIXTURE
        )
        assert len(transformed) == 2
        assert set(transformed["season_type"]) == {"Postseason"}, (
            "two WC rows transformed to "
            f"{sorted(set(transformed['season_type']))}. Before this plan every "
            "postseason game in project history read 'Regular'."
        )

    def test_the_postseason_rows_are_not_neutral(self) -> None:
        ingester = ingest_games.GameDataIngester()
        transformed = ingester.transform_schedule_data(
            season_2026.WEEK_19_POSTSEASON_FIXTURE
        )
        assert not transformed["neutral_site"].any()

    def test_the_postseason_rows_carry_their_stadium_ids(self) -> None:
        ingester = ingest_games.GameDataIngester()
        transformed = ingester.transform_schedule_data(
            season_2026.WEEK_19_POSTSEASON_FIXTURE
        )
        assert list(transformed["stadium_id"]) == ["BUF00", "DAL00"]


class TestTheColumnsSurviveSchemaValidation:
    """A column that the schema drops never reaches silver (the 28-06 trap)."""

    def test_game_schema_declares_stadium_id(self) -> None:
        from data.schemas import GameSchema

        assert "stadium_id" in GameSchema.model_fields, (
            "GameSchema has no stadium_id field. validate_bronze_to_silver calls "
            "model_dump(), so an undeclared column is SILENTLY DROPPED between the "
            "transform and the silver write -- gate-passing while never arriving."
        )

    def test_the_three_columns_survive_the_quality_gate(self) -> None:
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import GameSchema

        ingester = ingest_games.GameDataIngester()
        transformed = ingester.transform_schedule_data(
            season_2026.WEEK_19_POSTSEASON_FIXTURE
        )
        validated = validate_bronze_to_silver(transformed, GameSchema)

        for column in phase33_state.PLAN_33_06_IDENTITY_COLUMNS:
            assert column in validated.columns, (
                f"{column} did not survive validate_bronze_to_silver."
            )
        assert list(validated["stadium_id"]) == ["BUF00", "DAL00"]
        assert set(validated["season_type"]) == {"Postseason"}

    def test_the_recorded_silver_widths_bracket_the_change(self) -> None:
        """15 -> 18: the columns this plan adds, counted rather than described."""
        assert (
            phase33_state.SILVER_GAMES_COLUMNS_BEFORE
            + len(phase33_state.PLAN_33_06_IDENTITY_COLUMNS)
        ) == phase33_state.SILVER_GAMES_COLUMNS_AFTER


class TestTheLiveConsumerStillWorks:
    """`utils/similar_games.py` reads season_type today and gets a constant."""

    def test_the_consumer_now_sees_a_real_partition(self) -> None:
        from utils.similar_games import SimilarGamesEngine

        assert hasattr(SimilarGamesEngine, "_calculate_context_similarity"), (
            "the consumer that reads season_type has moved; re-point this control."
        )
        ingester = ingest_games.GameDataIngester()
        postseason = ingester.transform_schedule_data(
            season_2026.WEEK_19_POSTSEASON_FIXTURE
        )
        _require_capture()
        regular = season_2026.transform_captured_schedule()

        both = pd.concat([regular, postseason], ignore_index=True)
        assert set(both["season_type"]) == {"Regular", "Postseason"}, (
            "the two halves of the partition must both be reachable; a consumer "
            "reading season_type used to receive the constant 'Regular'."
        )
