"""The emergency-schedule-move candidate manifest equals its enumerator (Plan 33.2-10 Task 1).

SPEC R8, D33.2-04, D33.2-21; threats T-33.2-10-03, -04, -09, -10.

What is asserted, and why:

* the committed manifest equals a FRESH enumerator run, row for row, so neither a
  hand-added nor a hand-dropped row passes;
* every ``lock_utc`` equals ``utils.game_lock.game_lock`` of that game's silver kickoff,
  recomputed here rather than trusted from the script;
* the closest call (``2010_W14_NYG@MIN``) is present, the 2025 international games are
  absent, every D33.2-04 named event resolves to a manifest row, and the two week-move
  blind spots (Ike, Irma) are rows carrying ONLY a named-event criterion;
* the entry path works (a planted research entry appears) and refuses an entry with no
  source; the Tuesday/Wednesday scan reads the weekday in America/New_York (a Monday-night
  game stored as a UTC Tuesday is NOT surfaced); and a non-vacuity control plants a
  synthetic Tuesday game and requires the scan to find it.

The real-data tests read production silver READ-ONLY and skip when it is not built.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_schedule_move_candidates as bsmc
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"

CLOSEST_CALL = "2010_W14_NYG@MIN"
IKE = "2008_W10_BAL@HOU"
IRMA = "2017_W11_TB@MIA"
COVID_SUNDAY_TO_MONDAY = ("2020_W04_NE@KC", "2021_W15_LV@CLE")

#: Every event D33.2-04 names. Pinned here so a key dropped from the script fails.
D33_2_04_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "katrina_2005",
        "ike_2008",
        "metrodome_2010_roof_collapse",
        "metrodome_2010_tcf_bank",
        "buffalo_snow_2014",
        "irma_2017",
        "glendale_49ers_2020",
        "covid_2020_2021",
        "ida_2021",
        "buffalo_snow_2022",
    }
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def silver_games() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        pytest.skip("silver games is not built on this checkout")
    return pd.read_parquet(GAMES_PATH)


@pytest.fixture(scope="module")
def fresh_candidates(silver_games: pd.DataFrame) -> list[bsmc.Candidate]:
    return bsmc.build_from_data_root(REPO_ROOT / "data")


@pytest.fixture(scope="module")
def manifest_rows() -> list[dict[str, object]]:
    return bsmc.read_manifest_rows()


@pytest.fixture(scope="module")
def manifest_by_id(
    manifest_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {str(r["game_id"]): r for r in manifest_rows}


def _game(
    game_id: str,
    kickoff_utc: str,
    home: str,
    stadium: str,
    *,
    season: int = 2030,
    week: int = 1,
    neutral: bool = False,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "kickoff_et": pd.Timestamp(kickoff_utc, tz="UTC"),
        "home_team": home,
        "away_team": "AAA",
        "neutral_site": neutral,
        "stadium_id": stadium,
        "game_type": "REG",
    }


def _synthetic_games(extra: list[dict[str, object]] | None = None) -> pd.DataFrame:
    """Three ordinary Sunday home games for HHH at HHH00, plus *extra* rows."""
    rows = [
        _game("2030_W01_AAA@HHH", "2030-09-08T17:00:00", "HHH", "HHH00", week=1),
        _game("2030_W02_BBB@HHH", "2030-09-15T17:00:00", "HHH", "HHH00", week=2),
        _game("2030_W03_CCC@HHH", "2030-09-22T17:00:00", "HHH", "HHH00", week=3),
    ]
    rows.extend(extra or [])
    return pd.DataFrame(rows)


SYNTHETIC_COUNTRIES = {"HHH00": "USA", "OTH00": "USA", "LON00": "United Kingdom"}


def _enumerate(games: pd.DataFrame, **kwargs) -> dict[str, bsmc.Candidate]:
    kwargs.setdefault("named_events", {})
    kwargs.setdefault("research_discovered", {})
    kwargs.setdefault("international_ids", frozenset())
    out = bsmc.enumerate_candidates(
        games, SYNTHETIC_COUNTRIES, kwargs.pop("international_ids"), **kwargs
    )
    return {c.game_id: c for c in out}


# ------------------------------------------------ the manifest is the enumerator


class TestTheManifestIsTheEnumeratorsOutput:
    def test_the_game_id_sets_are_equal(self, fresh_candidates, manifest_rows) -> None:
        fresh = {c.game_id for c in fresh_candidates}
        committed = {str(r["game_id"]) for r in manifest_rows}
        assert committed - fresh == set(), (
            "rows in the manifest the enumerator does not produce"
        )
        assert fresh - committed == set(), (
            "enumerated candidates missing from the manifest"
        )

    def test_every_row_equals_the_fresh_row(
        self, fresh_candidates, manifest_by_id
    ) -> None:
        mismatched = [
            c.game_id
            for c in fresh_candidates
            if manifest_by_id.get(c.game_id) != c.as_row()
        ]
        assert mismatched == []

    def test_the_file_is_byte_identical_to_a_fresh_render(
        self, fresh_candidates
    ) -> None:
        rendered = bsmc.render_manifest(fresh_candidates)
        assert bsmc.MANIFEST_PATH.read_text(encoding="utf-8") == rendered

    def test_the_manifest_is_not_empty(self, manifest_rows) -> None:
        assert len(manifest_rows) > 0


class TestEveryLockIsTheLockRule:
    def test_lock_utc_equals_game_lock_of_the_silver_kickoff(
        self, silver_games, manifest_rows
    ) -> None:
        kickoffs = silver_games.set_index("game_id")["kickoff_et"]
        drifted = []
        for row in manifest_rows:
            game_id = str(row["game_id"])
            expected = pd.Timestamp(
                game_lock(kickoffs[game_id], game_id=game_id)
            ).tz_convert("UTC")
            if pd.Timestamp(str(row["lock_utc"])) != expected:
                drifted.append(game_id)
        assert drifted == []

    def test_every_row_carries_a_lock(self, manifest_rows) -> None:
        assert [r["game_id"] for r in manifest_rows if not r.get("lock_utc")] == []


# ------------------------------------------------------------------- named rows


class TestTheNamedRows:
    def test_the_closest_call_is_present(self, manifest_by_id) -> None:
        assert CLOSEST_CALL in manifest_by_id

    def test_the_2025_international_games_are_absent(self, manifest_by_id) -> None:
        international = bsmc.load_international_game_ids()
        assert len(international) == 7
        assert sorted(international & set(manifest_by_id)) == []

    def test_the_script_covers_exactly_the_d33_2_04_events(self) -> None:
        assert set(bsmc.NAMED_EVENT_GAME_IDS) == D33_2_04_EVENT_KEYS

    @pytest.mark.parametrize("event_key", sorted(D33_2_04_EVENT_KEYS))
    def test_each_named_event_resolves_to_a_manifest_row(
        self, event_key: str, manifest_by_id
    ) -> None:
        game_ids = bsmc.NAMED_EVENT_GAME_IDS[event_key]
        assert game_ids, f"{event_key} maps to no game id"
        for game_id in game_ids:
            row = manifest_by_id.get(game_id)
            assert row is not None, f"{event_key}: {game_id} is not a manifest row"
            assert f"named_event:{event_key}" in row["criteria"]

    @pytest.mark.parametrize("game_id", [IKE, IRMA])
    def test_the_week_moves_are_rows_with_no_mechanical_criterion(
        self, game_id: str, manifest_by_id
    ) -> None:
        criteria = manifest_by_id[game_id]["criteria"]
        assert any(str(c).startswith("named_event:") for c in criteria)
        assert not set(criteria) & set(bsmc.MECHANICAL_CRITERIA)

    @pytest.mark.parametrize("game_id", COVID_SUNDAY_TO_MONDAY)
    def test_the_covid_sunday_to_monday_moves_are_named_only(
        self, game_id: str, manifest_by_id
    ) -> None:
        assert not set(manifest_by_id[game_id]["criteria"]) & set(
            bsmc.MECHANICAL_CRITERIA
        )

    def test_the_week_move_blind_spot_counts_ike_and_irma(
        self, fresh_candidates
    ) -> None:
        assert bsmc.week_move_blindspot(fresh_candidates) >= 2

    def test_no_named_event_is_unresolved(self, fresh_candidates) -> None:
        assert bsmc.unresolved_named_events(fresh_candidates) == []


class TestTheTimeZoneTrap:
    def test_a_real_monday_night_game_is_not_a_tuesday_candidate(
        self, silver_games, manifest_by_id
    ) -> None:
        kickoff = silver_games.set_index("game_id").at[CLOSEST_CALL, "kickoff_et"]
        assert kickoff.tz_convert("UTC").day_name() == "Tuesday"  # the trap is real
        assert manifest_by_id[CLOSEST_CALL]["kickoff_weekday_et"] == "Monday"
        assert "tue_wed" not in manifest_by_id[CLOSEST_CALL]["criteria"]

    def test_a_synthetic_monday_night_game_is_not_surfaced(self) -> None:
        # Monday 20:15 ET on 2030-09-30 is Tuesday 00:15 UTC.
        monday_night = _game(
            "2030_W04_DDD@HHH", "2030-10-01T00:15:00", "HHH", "HHH00", week=4
        )
        found = _enumerate(_synthetic_games([monday_night]))
        assert "2030_W04_DDD@HHH" not in found


# ------------------------------------------------------------- non-vacuity controls


class TestTheScansFindWhatIsPlanted:
    def test_a_planted_tuesday_game_is_surfaced(self) -> None:
        tuesday = _game(
            "2030_W04_EEE@HHH", "2030-10-01T23:00:00", "HHH", "HHH00", week=4
        )
        found = _enumerate(_synthetic_games([tuesday]))
        assert found["2030_W04_EEE@HHH"].criteria == ("tue_wed",)

    def test_a_planted_domestic_neutral_game_is_surfaced(self) -> None:
        neutral = _game(
            "2030_W04_FFF@HHH",
            "2030-09-29T17:00:00",
            "HHH",
            "OTH00",
            week=4,
            neutral=True,
        )
        found = _enumerate(_synthetic_games([neutral]))
        assert found["2030_W04_FFF@HHH"].criteria == ("neutral_domestic",)

    def test_a_planted_foreign_neutral_game_is_not_surfaced(self) -> None:
        london = _game(
            "2030_W04_GGG@HHH",
            "2030-09-29T13:30:00",
            "HHH",
            "LON00",
            week=4,
            neutral=True,
        )
        assert "2030_W04_GGG@HHH" not in _enumerate(_synthetic_games([london]))

    def test_a_planted_unflagged_away_game_is_surfaced(self) -> None:
        away = _game("2030_W04_HHH2@HHH", "2030-09-29T17:00:00", "HHH", "OTH00", week=4)
        found = _enumerate(_synthetic_games([away]))
        assert found["2030_W04_HHH2@HHH"].criteria == ("away_unflagged",)

    def test_ordinary_games_surface_nothing(self) -> None:
        assert _enumerate(_synthetic_games()) == {}

    def test_an_international_correction_is_removed_from_the_scans(self) -> None:
        tuesday = _game(
            "2030_W04_EEE@HHH", "2030-10-01T23:00:00", "HHH", "HHH00", week=4
        )
        found = _enumerate(
            _synthetic_games([tuesday]),
            international_ids=frozenset({"2030_W04_EEE@HHH"}),
        )
        assert found == {}


class TestTheEntryPaths:
    def test_a_planted_research_entry_appears_in_the_regenerated_set(self) -> None:
        planted = {
            "2030_W02_BBB@HHH": {"reason": "moved a day", "source_url": "https://x"}
        }
        found = _enumerate(_synthetic_games(), research_discovered=planted)
        assert found["2030_W02_BBB@HHH"].criteria == ("research_discovered",)

    def test_a_planted_research_entry_appears_in_the_real_regeneration(
        self, silver_games
    ) -> None:
        planted = dict(bsmc.RESEARCH_DISCOVERED_GAME_IDS)
        planted["2019_W01_PIT@NE"] = {"reason": "planted", "source_url": "https://x"}
        out = bsmc.enumerate_candidates(
            silver_games,
            bsmc.load_venue_countries(REPO_ROOT / "data"),
            bsmc.load_international_game_ids(),
            research_discovered=planted,
        )
        assert (
            "research_discovered"
            in {c.game_id: c for c in out}["2019_W01_PIT@NE"].criteria
        )

    def test_a_named_game_is_unioned_unconditionally(self) -> None:
        found = _enumerate(
            _synthetic_games(), named_events={"storm": ("2030_W03_CCC@HHH",)}
        )
        assert found["2030_W03_CCC@HHH"].criteria == ("named_event:storm",)

    def test_a_research_entry_without_a_source_is_refused(self) -> None:
        with pytest.raises(bsmc.CandidateInputError, match="source_url"):
            _enumerate(
                _synthetic_games(),
                research_discovered={
                    "2030_W02_BBB@HHH": {"reason": "x", "source_url": ""}
                },
            )

    def test_a_committed_id_absent_from_silver_is_refused(self) -> None:
        with pytest.raises(bsmc.CandidateInputError, match="absent from silver"):
            _enumerate(
                _synthetic_games(), named_events={"storm": ("2030_W09_ZZZ@HHH",)}
            )

    def test_a_committed_id_naming_an_international_game_is_refused(self) -> None:
        with pytest.raises(bsmc.CandidateInputError, match="international"):
            _enumerate(
                _synthetic_games(),
                named_events={"storm": ("2030_W03_CCC@HHH",)},
                international_ids=frozenset({"2030_W03_CCC@HHH"}),
            )

    def test_a_stadium_with_no_venue_record_is_refused(self) -> None:
        stray = _game("2030_W04_EEE@HHH", "2030-09-29T17:00:00", "HHH", "NOPE0", week=4)
        with pytest.raises(bsmc.CandidateInputError, match="NOPE0"):
            _enumerate(_synthetic_games([stray]))


class TestTheLimitationIsStated:
    def test_the_script_docstring_states_the_week_move_blind_spot(self) -> None:
        doc = inspect.getdoc(bsmc) or ""
        for phrase in ("ORIGINAL schedule", "Ike", "Irma", "LATER WEEK"):
            assert phrase in doc

    def test_the_manifest_header_states_the_week_move_blind_spot(self) -> None:
        text = bsmc.MANIFEST_PATH.read_text(encoding="utf-8")
        header = text.split("[[candidates]]", 1)[0]
        for phrase in ("ORIGINAL schedule", "Ike", "Irma", "later WEEK"):
            assert phrase in header

    def test_the_manifest_parses_as_toml(self) -> None:
        parsed = tomllib.loads(bsmc.MANIFEST_PATH.read_text(encoding="utf-8"))
        assert parsed["candidate_count"] == len(parsed["candidates"])
