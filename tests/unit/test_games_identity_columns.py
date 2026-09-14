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

import json
import re
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Plan 33-12 Task 2(b): THE WHOLE MIGRATED STORE, not a sample.
#
# Plan 33-06 proved the DERIVATIONS above. It deliberately asserted nothing about
# the stored values, because at the time the store was still wrong on all 6,499
# rows -- `tests.phase33_state.POSTSEASON_PARTITION_EVIDENCE_NOTE` says exactly
# that and names this plan as the one that makes the column true. These are the
# assertions that became possible once it did.
#
# EVERY ROW, NOT A SAMPLE. The never-disagree property is asserted over the whole
# store and NAMES the offending game_ids on failure, because "some row disagrees"
# sends the reader back to the parquet to find out which.
#
# THE SKIP IS PINNED AND SPECIFIC. `data/` is gitignored, so a fresh checkout has
# no store at all; a hard failure there would be a false alarm about the checkout
# rather than a finding about the data. The skip distinguishes the two cases -- no
# file, versus a file the migration has not been run against -- so the message
# tells the reader which one they are in.
# ---------------------------------------------------------------------------

SILVER_GAMES_PATH = Path("data") / "silver" / "games.parquet"

PRIMARY_KEY_COLUMNS = ("game_id", "season", "week", "home_team", "away_team")


def _migrated_store() -> pd.DataFrame:
    """The live migrated silver frame, or a pinned skip naming which case applies."""
    if not SILVER_GAMES_PATH.exists():
        pytest.skip(
            "data/silver/games.parquet is absent. data/ is gitignored, so a fresh "
            "checkout carries no production store and there is nothing to assert "
            "about it. Run the Plan 33-12 migration -- 24 invocations of "
            "`uv run python scripts/ingest_games.py --season <YEAR>` for 2002-2025 "
            "-- to populate it."
        )
    frame = pd.read_parquet(SILVER_GAMES_PATH)
    if "stadium_id" not in frame.columns:
        pytest.skip(
            "data/silver/games.parquet exists but carries no stadium_id column, so "
            "the Plan 33-12 identity migration has not been run against this "
            "checkout. These assertions describe the MIGRATED store; asserting "
            "them against the pre-migration one would report a defect that was "
            "already known and already fixed."
        )
    return frame


class TestTheMigratedStoreCarriesRealIdentityColumns:
    """COLD-09 / D33-15 / D33-17, asserted on the store rather than the derivation."""

    def test_the_store_is_the_shape_the_migration_recorded(self) -> None:
        """Asserted first: the assertions below are vacuous against an empty frame."""
        frame = _migrated_store()
        assert len(frame) == phase33_state.SILVER_GAMES_ROWS_AFTER_IDENTITY
        assert len(frame.columns) == phase33_state.SILVER_GAMES_COLUMNS_AFTER_MEASURED
        first, last = phase33_state.SILVER_GAMES_SEASONS_AFTER_IDENTITY
        assert (int(frame["season"].min()), int(frame["season"].max())) == (first, last)

    def test_season_type_can_never_disagree_with_game_type_on_any_row(self) -> None:
        """D33-17, over ALL rows, naming the offenders.

        `season_type` is a FUNCTION of `game_type`, so there is no second source
        for the two to drift apart from -- which is a claim about the code. This
        is the claim about the STORE: that the rows actually written honour it.
        """
        frame = _migrated_store()
        disagreeing = [
            row["game_id"]
            for row in frame.to_dict("records")
            if row["season_type"] != ingest_games._derive_season_type(row)
        ]
        assert not disagreeing, (
            f"{len(disagreeing)} row(s) carry a season_type that disagrees with "
            f"their own game_type. First offenders: {disagreeing[:10]}. The column "
            "is a total function of game_type, so a disagreement means a row was "
            "written by something other than the derivation."
        )

    def test_the_partition_is_real_rather_than_a_constant(self) -> None:
        """The whole defect in one assertion: more than one distinct value exists."""
        frame = _migrated_store()
        counts = dict(frame["season_type"].value_counts())
        assert set(counts) == {"Regular", "Postseason"}, (
            f"season_type holds {sorted(counts)}. Before this migration it held "
            "only ['Regular'] -- on every WC, DIV, CON and SB game ever played."
        )
        assert dict(phase33_state.SILVER_SEASON_TYPE_COUNTS_AFTER_IDENTITY) == {
            key: int(value) for key, value in counts.items()
        }

    def test_the_postseason_population_is_at_least_ninety_rows(self) -> None:
        """D33-17's floor, and the recorded count, checked against each other."""
        frame = _migrated_store()
        measured = int((frame["season_type"] != "Regular").sum())
        assert measured >= 90, (
            f"only {measured} postseason rows exist across 2002-2025; the phase's "
            "own floor is 90 and the measured value at migration time was "
            f"{phase33_state.POSTSEASON_ROW_COUNT}."
        )
        assert measured == phase33_state.POSTSEASON_ROW_COUNT

    def test_neutral_site_is_no_longer_a_constant_false(self) -> None:
        frame = _migrated_store()
        measured = int(frame["neutral_site"].sum())
        assert measured == phase33_state.NEUTRAL_SITE_TRUE_ROW_COUNT, (
            f"{measured} rows read neutral_site True; the migration measured "
            f"{phase33_state.NEUTRAL_SITE_TRUE_ROW_COUNT}. These are the same games "
            "HISTORICAL_NEUTRAL_MISRESOLUTION enumerates."
        )
        assert measured > 0

    def test_stadium_id_is_present_on_every_row(self) -> None:
        """A presence RATE, not a presence check: a partial re-ingest is the risk."""
        frame = _migrated_store()
        present = int(frame["stadium_id"].notna().sum())
        assert present == len(frame), (
            f"stadium_id is null on {len(frame) - present} of {len(frame)} rows. "
            "Since D33.1-06 the contextual builder routes EVERY game by its own "
            "stadium_id, so a null is a hard build failure downstream, not a gap."
        )

    def test_no_primary_key_column_is_null_anywhere(self) -> None:
        """Cheap, and it catches a partial re-ingest a row count alone would not."""
        frame = _migrated_store()
        nulls = {
            column: int(frame[column].isna().sum()) for column in PRIMARY_KEY_COLUMNS
        }
        assert not any(nulls.values()), f"null primary-key cells: {nulls}"
        assert not frame["game_id"].duplicated().any(), (
            "duplicate game_ids exist, so the latest-wins upsert did not key "
            "cleanly -- two ingests wrote the same game under different ids."
        )

    def test_the_recorded_integrity_checks_still_hold(self) -> None:
        """The manifest's SILVER_GAMES_INTEGRITY_AFTER_IDENTITY, re-measured."""
        frame = _migrated_store()
        measured = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "distinct_seasons": frame["season"].nunique(),
            "duplicate_game_ids": int(frame["game_id"].duplicated().sum()),
            "stadium_id_nulls": int(frame["stadium_id"].isna().sum()),
            "primary_key_nulls": sum(
                int(frame[column].isna().sum()) for column in PRIMARY_KEY_COLUMNS
            ),
        }
        recorded = dict(phase33_state.SILVER_GAMES_INTEGRITY_AFTER_IDENTITY)
        for key, value in measured.items():
            assert value == recorded[key], (
                f"{key} is {value} but the migration recorded {recorded[key]}"
            )


class TestTheRecordedSilverWidthIsSixteenAndSaysWhy:
    """The correction to Plan 33-06's 18, appended rather than edited.

    `SILVER_GAMES_COLUMNS_AFTER` is 18 and is left exactly as written -- the
    manifest is append-only and a slot records what was believed when it was
    measured. `TestTheColumnsSurviveSchemaValidation.test_the_recorded_silver_widths_bracket_the_change`
    above still asserts 15 + 3 == 18, which is a true statement about the two
    constants and a FALSE one about the store. These tests say so out loud rather
    than leaving the discrepancy for a later reader to trip over.
    """

    def test_only_one_of_the_three_identity_columns_was_ever_new(self) -> None:
        """The arithmetic error, stated as the measurement that exposes it."""
        assert set(phase33_state.PLAN_33_06_IDENTITY_COLUMNS) == {
            "stadium_id",
            "neutral_site",
            "season_type",
        }
        frame = _migrated_store()
        assert {"season_type", "neutral_site"}.issubset(frame.columns)
        assert (
            phase33_state.SILVER_GAMES_COLUMNS_BEFORE + 1
            == phase33_state.SILVER_GAMES_COLUMNS_AFTER_MEASURED
        ), (
            "15 + 1 = 16. season_type and neutral_site were already COLUMNS before "
            "the migration; what they were not was TRUE. Only stadium_id is new."
        )

    def test_the_correction_states_the_superseded_value(self) -> None:
        """A correction nobody can find is not a correction."""
        text = phase33_state.SILVER_GAMES_COLUMNS_AFTER_CORRECTION
        assert "SILVER_GAMES_COLUMNS_AFTER is 18" in text
        assert "16" in text
        assert phase33_state.SILVER_GAMES_COLUMNS_AFTER == 18, (
            "the superseded slot was EDITED. The manifest is append-only: a slot "
            "is a record of what was believed, and correcting it in place destroys "
            "the record the correction exists to explain."
        )


class TestTheMigrationsDeclaredBlastRadiusIsIntact:
    """T-33-61: the declaration must not be amendable after the fact."""

    def test_the_declaration_names_silver_and_bronze_and_not_the_duckdb(self) -> None:
        declared = list(phase33_state.IDENTITY_MIGRATION_EXPECTED_CHANGED_FILES)
        assert any("silver/games.parquet" in entry for entry in declared)
        assert any("bronze" in entry for entry in declared)
        assert not any("nfl_predictions.duckdb" in entry for entry in declared), (
            "the DuckDB is deliberately absent because upsert_silver bypasses "
            "save_dataframe; see the manifest block for the read finding."
        )
        assert (
            "nfl_predictions.duckdb"
            in phase33_state.IDENTITY_MIGRATION_EXPLICITLY_NOT_EXPECTED
        ), "the absence must be a CLAIM, not a gap"

    def test_every_season_names_the_immutable_bytes_that_reproduce_it(self) -> None:
        """Codex MEDIUM: a migration whose input cannot be named is not reproducible."""
        mapping = phase33_state.IDENTITY_MIGRATION_PIN_MAPPING
        seasons_by_dataset: dict[str, set[int]] = {}
        for season, dataset, path, digest in mapping:
            seasons_by_dataset.setdefault(dataset, set()).add(season)
            assert path.startswith("bronze/")
            assert len(digest) == 64
        for dataset in ("schedules", "pbp"):
            assert seasons_by_dataset[dataset] == set(range(2002, 2026)), (
                f"{dataset} does not cover 2002-2025; a season with no covering "
                "pin must be REFUSED by name, never re-ingested from whatever the "
                "upstream happens to serve today."
            )

    def test_the_expected_bronze_count_matches_one_file_per_season(self) -> None:
        assert (
            len(range(2002, 2026))
            == phase33_state.IDENTITY_MIGRATION_EXPECTED_BRONZE_COUNT
        )

    def test_the_bronze_pattern_matches_a_real_written_name_and_rejects_a_wrong_one(
        self,
    ) -> None:
        """The pattern is what the verification matched on, so it must be exercised."""
        pattern = re.compile(phase33_state.IDENTITY_MIGRATION_BRONZE_PATTERN)
        assert pattern.match("bronze/games_raw_bronze_2002_W00_20260914T064324.parquet")
        assert not pattern.match(
            "bronze/schedules_raw_bronze_2002_W00_20260905T044449.parquet"
        ), "the pattern must not match the PINNED snapshots the ingest reads"
        assert not pattern.match(
            "bronze/games_raw_bronze_2026_W00_20260914T064324.parquet"
        ), "2026 is outside this migration's declared season range"


class TestTheFourthMovedColumnWasDeclaredNotDiscovered:
    """venue_roof, and why it is inside this migration's cause rather than beside it."""

    def test_the_roof_movement_is_recorded_with_its_per_venue_breakdown(self) -> None:
        movement = phase33_state.IDENTITY_MIGRATION_VENUE_ROOF_MOVEMENT
        assert movement, "the breakdown is the evidence; a bare count is a claim"
        assert (
            sum(rows for *_, rows in movement)
            == phase33_state.IDENTITY_MIGRATION_VENUE_ROOF_ROWS_MOVED
        ), "the per-venue rows must sum to the recorded total"
        for stadium_id, _venue, before, after, rows in movement:
            assert before != after and rows > 0
            assert stadium_id in _venues_by_stadium_id(), (
                f"{stadium_id} is not in data/venues.json, so the stadium_id key "
                "could not have produced the corrected roof for it"
            )

    def test_the_migrated_roof_distribution_matches_the_record(self) -> None:
        frame = _migrated_store()
        measured = {
            key: int(value) for key, value in frame["venue_roof"].value_counts().items()
        }
        assert measured == dict(
            phase33_state.SILVER_VENUE_ROOF_COUNTS_AFTER_IDENTITY
        ), f"roof distribution is {measured}"


def _venues_by_stadium_id() -> set[str]:
    venues_path = Path("data") / "venues.json"
    if not venues_path.exists():
        pytest.skip("data/venues.json is absent")
    with open(venues_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        venue["stadium_id"]
        for venue in payload.get("venues", [])
        if venue.get("stadium_id")
    }
