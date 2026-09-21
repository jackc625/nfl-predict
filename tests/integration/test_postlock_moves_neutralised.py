"""A game moved by an emergency announced after its lock is built with its pre-move facts.

Plan 33.2-10 Task 4 (SPEC R8, D33.2-04, D33.2-21; threats T-33.2-10-05, -06).

What is asserted, and why:

* THE ACCESSOR (``features.schedule_moves.facts_at_lock``) on planted tables: a post-lock
  move reverts to the pre-move facts; an all-pre-lock game keeps its actual facts; a game
  moved twice resolves to the facts after its LAST pre-lock move; a game with no row and a
  ``scheduled_not_moved`` row under either verdict change nothing; date and week moves
  revert on the same terms; a revert whose ``to_value`` is not what the row records is
  REFUSED rather than neutralised to facts that were never true.
* THE LOADER parses once per process (a spy counts the parse across a whole season),
  refuses a malformed table at load, and refuses a published post-lock block that
  disagrees with the rows.
* THE REAL TABLE AGAINST SILVER: the post-lock set of REAL moves is non-empty (so neither an
  empty set nor a set of no-op scheduled rows satisfies the loops trivially); every
  pre-lock game resolves to its actual facts; and every move chain in the table reverts
  cleanly against silver even if every verdict were post-lock, which exercises the date
  and week reverts on real rows.
* THE BUILDERS: for every post-lock game, the contextual builder's venue, roof, travel,
  weekday and surface columns equal those of the same game built at its pre-move facts,
  and the weather-location resolution (``scripts.ingest_weather``) names the pre-move
  venue. For the closest call (a pre-lock game) both equal the actual facts.
* GOLD, row by row against the before-gold copy taken for rung 3: the post-lock game moved
  in EXACTLY the contextual columns whose raw value differs between its actual and its
  pre-move facts.

The real-data halves read production stores READ-ONLY and skip when absent.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from features import schedule_moves
from features.contextual import ContextualFeaturesCalculator
from features.schedule_moves import (
    ScheduleMoveTableError,
    facts_at_lock,
    load_schedule_moves,
    post_lock_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"
GOLD_BEFORE_DIR = REPO_ROOT / "outputs" / "p332_rung3_gold_before"
GOLD_AFTER_DIR = REPO_ROOT / "data" / "gold"
GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")

CLOSEST_CALL = "2010_W14_NYG@MIN"
ID_COLUMNS = ("game_id", "season", "week", "home_team", "away_team")

# The contextual columns a schedule fact reaches: the venue family (venue, roof, surface,
# travel, time zone) and the weekday family. The rest family is compared too, because a
# kickoff revert would reach it.
SCHEDULE_FACT_COLUMNS: tuple[str, ...] = (
    "away_travel_distance_miles",
    "away_timezone_diff_hours",
    "away_abs_timezone_diff_hours",
    "away_travel_fatigue_score",
    "away_cross_country_travel",
    "away_eastward_travel",
    "away_westward_travel",
    "thursday_game",
    "monday_game",
    "saturday_game",
    "short_week",
    "game_day_of_week",
    "venue_outdoor",
    "venue_indoor",
    "venue_retractable",
    "venue_elevation_ft",
    "venue_high_altitude",
    "venue_cold_climate",
    "venue_warm_climate",
    "venue_capacity",
    "venue_large_stadium",
    "home_rest_days",
    "away_rest_days",
    "surface_mismatch",
    "season_progress",
    "late_season",
)

HEADER = "# planted move table\n"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "2030_W05_AAA@HHH",
        "move_index": 1,
        "what_moved": "venue",
        "from_value": "HHH00",
        "to_value": "OTH00",
        "announced_at_utc": "2030-10-05T22:00:00Z",
        "source_url": "https://example.invalid/story",
        "source_title": "planted",
        "source_published_at": "2030-10-05",
        "lock_utc": "2030-10-05T22:00:00Z",
        "verdict": "pre_lock",
        "notes": "planted",
    }
    row.update(overrides)
    return row


def _write_table(
    tmp_path: Path, rows: list[dict[str, object]], summary: str = ""
) -> Path:
    blocks = []
    for row in rows:
        lines = ["[[moves]]"]
        for key, value in row.items():
            rendered = str(value) if isinstance(value, int) else json.dumps(value)
            lines.append(f"{key} = {rendered}")
        blocks.append("\n".join(lines))
    path = tmp_path / f"moves_{len(list(tmp_path.iterdir()))}.toml"
    path.write_text(HEADER + "\n\n".join(blocks) + "\n" + summary, encoding="utf-8")
    return path


# 2030-10-06 17:00 UTC = 13:00 ET on Sunday 2030-10-06; the planted lock is the day before.
GAME = {
    "game_id": "2030_W05_AAA@HHH",
    "stadium_id": "OTH00",
    "kickoff_et": pd.Timestamp("2030-10-06 17:00", tz="UTC"),
    "week": 5,
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    schedule_moves._load_cached.cache_clear()
    yield
    schedule_moves._load_cached.cache_clear()


# ------------------------------------------------------------------ the accessor


class TestTheAccessorOnPlantedTables:
    def test_a_post_lock_venue_move_reverts_to_the_pre_move_venue(
        self, tmp_path
    ) -> None:
        table = _write_table(
            tmp_path, [_row(verdict="post_lock", source_url="", announced_at_utc="")]
        )
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.stadium_id == "HHH00"
        assert facts.weather_stadium_id == "HHH00"
        assert facts.neutralised
        assert facts.kickoff_et.date().isoformat() == "2030-10-06"

    def test_an_all_pre_lock_game_keeps_its_actual_facts(self, tmp_path) -> None:
        table = _write_table(tmp_path, [_row()])
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.stadium_id == "OTH00"
        assert not facts.neutralised

    def test_an_announcement_exactly_at_the_lock_is_known(self, tmp_path) -> None:
        table = _write_table(tmp_path, [_row(announced_at_utc="2030-10-05T22:00:00Z")])
        assert facts_at_lock(GAME["game_id"], GAME, table_path=table).stadium_id == (
            "OTH00"
        )

    def test_two_moves_resolve_to_the_facts_after_the_last_pre_lock_one(
        self, tmp_path
    ) -> None:
        """Move 1 (pre-lock) HHH00 -> MID00; move 2 (post-lock) MID00 -> OTH00."""
        table = _write_table(
            tmp_path,
            [
                _row(move_index=1, from_value="HHH00", to_value="MID00"),
                _row(
                    move_index=2,
                    from_value="MID00",
                    to_value="OTH00",
                    verdict="post_lock",
                    source_url="",
                    announced_at_utc="",
                ),
            ],
        )
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.stadium_id == "MID00"
        assert [m.move_index for m in facts.neutralised_moves] == [2]

    def test_a_game_with_no_row_is_unchanged(self, tmp_path) -> None:
        table = _write_table(tmp_path, [_row(game_id="2030_W05_BBB@CCC")])
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert (facts.stadium_id, facts.week) == ("OTH00", 5)
        assert facts.neutralised_moves == ()

    @pytest.mark.parametrize("verdict", ["pre_lock", "post_lock"])
    def test_a_scheduled_row_changes_nothing_under_either_verdict(
        self, tmp_path, verdict
    ) -> None:
        table = _write_table(
            tmp_path,
            [
                _row(
                    what_moved="scheduled_not_moved",
                    from_value="OTH00 2030-10-06",
                    to_value="OTH00 2030-10-06",
                    verdict=verdict,
                )
            ],
        )
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.stadium_id == "OTH00"
        assert facts.kickoff_et.date().isoformat() == "2030-10-06"
        assert not facts.neutralised

    def test_a_post_lock_date_move_reverts_the_date_and_keeps_the_clock(
        self, tmp_path
    ) -> None:
        table = _write_table(
            tmp_path,
            [
                _row(
                    what_moved="date",
                    from_value="2030-10-05",
                    to_value="2030-10-06",
                    verdict="post_lock",
                    source_url="",
                    announced_at_utc="",
                )
            ],
        )
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.kickoff_et.isoformat() == "2030-10-05T13:00:00-04:00"
        assert facts.kickoff_et.weekday() == 5  # Saturday, read in America/New_York

    def test_a_post_lock_week_move_reverts_week_and_date(self, tmp_path) -> None:
        table = _write_table(
            tmp_path,
            [
                _row(
                    what_moved="week",
                    from_value="W02 2030-09-15",
                    to_value="W05 2030-10-06",
                    verdict="post_lock",
                    source_url="",
                    announced_at_utc="",
                )
            ],
        )
        facts = facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert facts.week == 2
        assert facts.kickoff_et.date().isoformat() == "2030-09-15"

    def test_a_revert_that_does_not_land_on_the_row_is_refused(self, tmp_path) -> None:
        table = _write_table(
            tmp_path,
            [
                _row(
                    to_value="ZZZ00",
                    verdict="post_lock",
                    source_url="",
                    announced_at_utc="",
                )
            ],
        )
        with pytest.raises(ScheduleMoveTableError, match="disagree"):
            facts_at_lock(GAME["game_id"], GAME, table_path=table)


class TestTheLoader:
    def test_the_table_is_parsed_once_for_a_whole_season(
        self, tmp_path, monkeypatch
    ) -> None:
        table = _write_table(tmp_path, [_row()])
        calls: list[Path] = []
        original = schedule_moves._parse_schedule_move_table

        def spy(path: Path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(schedule_moves, "_parse_schedule_move_table", spy)
        for week in range(1, 19):
            for game in range(16):
                row = {**GAME, "game_id": f"2030_W{week:02d}_G{game:02d}@HHH"}
                facts_at_lock(row["game_id"], row, table_path=table)
        facts_at_lock(GAME["game_id"], GAME, table_path=table)
        assert len(calls) == 1

    def test_the_mapping_is_read_only_and_its_records_frozen(self, tmp_path) -> None:
        moves = load_schedule_moves(_write_table(tmp_path, [_row()]))
        with pytest.raises(TypeError):
            moves["x"] = ()  # type: ignore[index]
        with pytest.raises(AttributeError):
            moves[GAME["game_id"]][0].verdict = "post_lock"  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"lock_utc": "2030-10-05T22:00:00"}, "time zone"),
            ({"what_moved": "stadium"}, "outside"),
            ({"verdict": "maybe"}, "verdict"),
            ({"source_url": ""}, "source_url"),
            ({"announced_at_utc": "2030-10-05T22:00:01Z"}, "AT or before"),
            ({"to_value": "not-a-venue"}, "not a venue value"),
        ],
    )
    def test_a_malformed_row_is_refused_at_load(
        self, tmp_path, overrides, match
    ) -> None:
        with pytest.raises(ScheduleMoveTableError, match=match):
            load_schedule_moves(_write_table(tmp_path, [_row(**overrides)]))

    def test_a_published_block_that_disagrees_with_the_rows_is_refused(
        self, tmp_path
    ) -> None:
        summary = (
            '\n[post_lock_summary]\nreal_moves = ["2099_W01_X@Y"]\n'
            "scheduled_not_moved = []\n"
        )
        with pytest.raises(ScheduleMoveTableError, match="disagrees"):
            load_schedule_moves(_write_table(tmp_path, [_row()], summary))


# -------------------------------------------------------- the real table and silver


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        pytest.skip("silver games is not built on this checkout")
    return pd.read_parquet(GAMES_PATH).set_index("game_id", drop=False)


def _real_post_lock_games() -> list[str]:
    return post_lock_summary()["real_moves"]


class TestTheRealTable:
    def test_the_post_lock_set_of_real_moves_is_not_empty(self) -> None:
        assert _real_post_lock_games(), (
            "no REAL move is post-lock; the neutralisation loops would pass vacuously"
        )

    def test_the_published_list_is_committed_and_separates_scheduled_games(
        self,
    ) -> None:
        import tomllib

        table = tomllib.loads(
            schedule_moves.SCHEDULE_MOVES_PATH.read_text(encoding="utf-8")
        )
        published = table[schedule_moves.POST_LOCK_SUMMARY_KEY]
        assert sorted(published["real_moves"]) == _real_post_lock_games()
        assert not set(published["real_moves"]) & set(published["scheduled_not_moved"])

    def test_every_post_lock_game_resolves_to_its_pre_move_facts(self, silver) -> None:
        for game_id in _real_post_lock_games():
            facts = facts_at_lock(game_id, silver.loc[game_id])
            assert facts.neutralised
            first_reverted = facts.neutralised_moves[-1]
            if first_reverted.what_moved == "venue":
                assert facts.stadium_id == first_reverted.from_value
                assert silver.at[game_id, "stadium_id"] == (
                    facts.neutralised_moves[0].to_value
                )

    def test_every_pre_lock_game_resolves_to_its_actual_facts(self, silver) -> None:
        post = set(_real_post_lock_games())
        for game_id in load_schedule_moves():
            if game_id in post:
                continue
            facts = facts_at_lock(game_id, silver.loc[game_id])
            assert facts.stadium_id == silver.at[game_id, "stadium_id"], game_id
            assert facts.week == silver.at[game_id, "week"], game_id
            assert not facts.neutralised, game_id

    def test_every_move_chain_reverts_cleanly_against_silver(
        self, silver, tmp_path
    ) -> None:
        """Flip every verdict to post-lock: each real chain must revert without refusal.

        This drives the date and week reverts over every real row, so a table whose
        ``from``/``to`` chain did not match silver would fail here, not in some later build.
        """
        import tomllib

        table = tomllib.loads(
            schedule_moves.SCHEDULE_MOVES_PATH.read_text(encoding="utf-8")
        )
        flipped = [
            {**row, "verdict": "post_lock", "source_url": "", "announced_at_utc": ""}
            for row in table["moves"]
        ]
        path = _write_table(tmp_path, flipped)
        reverted = 0
        for game_id in {str(row["game_id"]) for row in flipped}:
            facts = facts_at_lock(game_id, silver.loc[game_id], table_path=path)
            reverted += facts.neutralised
        assert reverted == len(
            {
                str(r["game_id"])
                for r in flipped
                if r["what_moved"] != "scheduled_not_moved"
            }
        )


# ----------------------------------------------------------------- the builders


def _contextual_row(calculator, season_games: pd.DataFrame, game_id: str) -> pd.Series:
    built = calculator.build_features(season_games, datetime(2026, 9, 21, tzinfo=UTC))
    return built.set_index("game_id").loc[game_id]


def _season_frame(silver: pd.DataFrame, season: int) -> pd.DataFrame:
    return silver[silver["season"] == season].reset_index(drop=True)


@pytest.fixture(scope="module")
def calculator() -> ContextualFeaturesCalculator:
    return ContextualFeaturesCalculator()


class TestTheBuildersUseThePreMoveFacts:
    """The actual game against the same game built at its pre-move facts."""

    @staticmethod
    def _at_facts(season_games: pd.DataFrame, game_id: str, facts) -> pd.DataFrame:
        probe = season_games.copy()
        position = probe.index[probe["game_id"] == game_id][0]
        probe.loc[position, "game_id"] = f"PROBE_{game_id}"
        probe.loc[position, "stadium_id"] = facts.stadium_id
        probe.loc[position, "kickoff_et"] = pd.Timestamp(facts.kickoff_et).tz_convert(
            probe["kickoff_et"].dt.tz
        )
        probe.loc[position, "week"] = facts.week
        return probe

    def test_every_post_lock_game_is_built_at_its_pre_move_facts(
        self, silver, calculator
    ) -> None:
        for game_id in _real_post_lock_games():
            season_games = _season_frame(silver, int(silver.at[game_id, "season"]))
            facts = facts_at_lock(game_id, silver.loc[game_id])
            actual = _contextual_row(calculator, season_games, game_id)
            expected = _contextual_row(
                calculator,
                self._at_facts(season_games, game_id, facts),
                f"PROBE_{game_id}",
            )
            for column in SCHEDULE_FACT_COLUMNS:
                assert actual[column] == expected[column], (game_id, column)

    def test_the_post_lock_game_no_longer_carries_its_post_move_venue(
        self, silver, calculator
    ) -> None:
        """Non-vacuity: at least one venue-family column differs from the as-played build."""
        for game_id in _real_post_lock_games():
            season_games = _season_frame(silver, int(silver.at[game_id, "season"]))
            as_played = season_games.copy()
            as_played.loc[as_played["game_id"] == game_id, "game_id"] = (
                f"PLAYED_{game_id}"
            )
            actual = _contextual_row(calculator, season_games, game_id)
            played = _contextual_row(calculator, as_played, f"PLAYED_{game_id}")
            differing = [c for c in SCHEDULE_FACT_COLUMNS if actual[c] != played[c]]
            assert differing, game_id

    def test_the_closest_call_is_built_at_its_actual_facts(
        self, silver, calculator
    ) -> None:
        season_games = _season_frame(silver, 2010)
        facts = facts_at_lock(CLOSEST_CALL, silver.loc[CLOSEST_CALL])
        assert (facts.stadium_id, facts.neutralised) == ("DET00", False)
        actual = _contextual_row(calculator, season_games, CLOSEST_CALL)
        expected = _contextual_row(
            calculator,
            self._at_facts(season_games, CLOSEST_CALL, facts),
            f"PROBE_{CLOSEST_CALL}",
        )
        for column in SCHEDULE_FACT_COLUMNS:
            assert actual[column] == expected[column], column

    def test_the_weather_location_is_the_pre_move_venue(self, silver) -> None:
        from scripts.ingest_weather import WeatherDataIngester

        ingester = WeatherDataIngester.__new__(WeatherDataIngester)
        venues = pd.DataFrame(
            json.loads((REPO_ROOT / "data" / "venues.json").read_text("utf-8"))[
                "venues"
            ]
        )
        for game_id in _real_post_lock_games():
            facts = facts_at_lock(game_id, silver.loc[game_id])
            venue = ingester._resolve_venue_record_for_game(silver.loc[game_id], venues)
            assert (
                venue["stadium_id"]
                == facts.weather_stadium_id
                != (silver.at[game_id, "stadium_id"])
            )
        closest = ingester._resolve_venue_record_for_game(
            silver.loc[CLOSEST_CALL], venues
        )
        assert closest["stadium_id"] == silver.at[CLOSEST_CALL, "stadium_id"]


# ------------------------------------------------------------------------- gold


needs_gold_copy = pytest.mark.skipif(
    not all((GOLD_BEFORE_DIR / f"{m}.parquet").is_file() for m in GOLD_MATRICES),
    reason=(
        "outputs/p332_rung3_gold_before/ (the before-gold copy taken for rung 3) is "
        "absent -- gitignored runtime evidence"
    ),
)


RUNG3_DOCUMENT = REPO_ROOT / "outputs" / "fingerprints" / "p332_rung3.json"


def _skip_unless_the_game_rows_are_rung_three(matrix: str, after: pd.DataFrame) -> None:
    """Skip once a later rung has moved the post-lock games' seasons past rung 3.

    The expected moved-column set is rung 3's own; a later rung (the day-before weather,
    for one) legitimately moves more columns on these rows, and judging that against
    rung 3's prediction would be judging the wrong rebuild.
    """
    import scripts.fingerprint_gold as fg

    if not RUNG3_DOCUMENT.is_file():
        pytest.skip(f"{RUNG3_DOCUMENT} is absent (gitignored runtime evidence)")
    seasons = {str(game_id[:4]) for game_id in _real_post_lock_games()}
    rung3 = json.loads(RUNG3_DOCUMENT.read_text(encoding="utf-8"))[matrix]["columns"]
    live = fg.fingerprint_matrix(after.reset_index())["columns"]
    for column, digests in rung3.items():
        if fg._is_build_clock(column) or column not in live:
            continue
        if any(live[column].get(s) != digests.get(s) for s in seasons):
            pytest.skip("live gold has moved on in the post-lock seasons since rung 3")


@needs_gold_copy
class TestGoldCarriesThePreMoveFacts:
    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_the_post_lock_game_moved_in_exactly_its_changed_fact_columns(
        self, matrix, silver, calculator
    ) -> None:
        before = pd.read_parquet(GOLD_BEFORE_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        after = pd.read_parquet(GOLD_AFTER_DIR / f"{matrix}.parquet").set_index(
            "game_id"
        )
        _skip_unless_the_game_rows_are_rung_three(matrix, after)
        for game_id in _real_post_lock_games():
            season_games = _season_frame(silver, int(silver.at[game_id, "season"]))
            as_played = season_games.copy()
            as_played.loc[as_played["game_id"] == game_id, "game_id"] = (
                f"PLAYED_{game_id}"
            )
            raw_now = _contextual_row(calculator, season_games, game_id)
            raw_played = _contextual_row(calculator, as_played, f"PLAYED_{game_id}")
            # Only the schedule-fact columns are compared raw: renaming the game to build
            # it as played also detaches it from the Elo schedule the spot flags read.
            raw_changed = {
                column
                for column in SCHEDULE_FACT_COLUMNS
                if column in before.columns and raw_now[column] != raw_played[column]
            }
            gold_changed = {
                column
                for column in before.columns
                if column not in ID_COLUMNS
                and column != "feature_timestamp"
                and not (
                    before.at[game_id, column] == after.at[game_id, column]
                    or (
                        pd.isna(before.at[game_id, column])
                        and pd.isna(after.at[game_id, column])
                    )
                )
            }
            assert raw_changed, game_id
            assert gold_changed == raw_changed, (game_id, sorted(gold_changed))
