"""The ingest applies the cited kickoff-hour corrections through ONE resolver.

Plan 33.2-12, p332_ extra step 3c. ``scripts.ingest_games.resolve_kickoff_gametime`` is
the single decision behind the clock a game's ``kickoff_et`` is built from; the ingest and
the one-shot store repair both call it. Rows are built in the captured feed's shape
(``tests.fixtures.season_2026``); no network.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import ingest_games
from tests.fixtures import season_2026

RECORDED = "2003_W08_MIA@LAC"  # Monday night, played in Tempe after the Cedar fire


def _feed_row(*, gameday: str, gametime: str) -> dict[str, object]:
    return {
        "game_id": "2003_08_MIA_SD",
        "season": 2003,
        "game_type": "REG",
        "week": 8,
        "gameday": gameday,
        "gametime": gametime,
        "away_team": "MIA",
        "home_team": "SD",
        "location": "Home",
        "stadium_id": "SDG00",
        "stadium": "Qualcomm Stadium",
        "roof": "outdoors",
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=list(season_2026.CAPTURED_FEED_COLUMNS))


def _kickoff(frame: pd.DataFrame) -> str:
    out = ingest_games.GameDataIngester().transform_schedule_data(frame)
    assert list(out["game_id"]) == [RECORDED]
    return pd.Timestamp(out.iloc[0]["kickoff_et"]).strftime("%Y-%m-%d %H:%M")


class TestTheRecordIsWhatTheIngestReads:
    def test_the_record_holds_68_cited_entries(self) -> None:
        corrections = ingest_games.load_kickoff_hour_corrections()
        assert len(corrections) == 68
        for game_id, correction in corrections.items():
            assert correction.source_url.startswith("http"), game_id
            assert correction.feed_gametime == "09:00", game_id
            assert correction.corrected_gametime == "21:00", game_id
            assert int(game_id[:4]) in (2002, 2003, 2004, 2005), game_id

    def test_the_loader_returns_an_immutable_mapping_parsed_once(self) -> None:
        corrections = ingest_games.load_kickoff_hour_corrections()
        assert corrections is ingest_games.load_kickoff_hour_corrections()
        with pytest.raises(TypeError):
            corrections[RECORDED] = None  # type: ignore[index]


class TestTheTransformAppliesTheCorrection:
    def test_the_feed_morning_clock_comes_out_at_nine_pm(self) -> None:
        assert _kickoff(
            _frame([_feed_row(gameday="2003-10-27", gametime="09:00")])
        ) == ("2003-10-27 21:00")

    def test_an_already_corrected_feed_row_comes_out_identical(self) -> None:
        """Idempotence: the day the feed fixes its own clock, nothing changes."""
        assert _kickoff(
            _frame([_feed_row(gameday="2003-10-27", gametime="21:00")])
        ) == ("2003-10-27 21:00")

    def test_a_third_clock_on_a_recorded_game_propagates(self) -> None:
        frame = _frame([_feed_row(gameday="2003-10-27", gametime="20:30")])
        with pytest.raises(ingest_games.KickoffCorrectionDriftError) as excinfo:
            ingest_games.GameDataIngester().transform_schedule_data(frame)
        assert RECORDED in str(excinfo.value) and "20:30" in str(excinfo.value)

    def test_a_moved_date_on_a_recorded_game_propagates(self) -> None:
        frame = _frame([_feed_row(gameday="2003-10-28", gametime="09:00")])
        with pytest.raises(ingest_games.KickoffCorrectionDriftError):
            ingest_games.GameDataIngester().transform_schedule_data(frame)

    def test_the_drift_error_has_its_own_re_raise_arm(self) -> None:
        assert not issubclass(
            ingest_games.KickoffCorrectionDriftError, ingest_games.IdentityColumnError
        )
        assert not issubclass(
            ingest_games.KickoffCorrectionDriftError,
            ingest_games.VenueOverrideDriftError,
        )

    def test_an_unrecorded_game_keeps_the_feed_clock(self) -> None:
        assert (
            ingest_games.resolve_kickoff_gametime(
                "2010_W10_DAL@GB", "2010-11-14", "09:00"
            )
            == "09:00"
        )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "kickoff_hour_corrections.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _entry(**overrides: str) -> str:
    fields = {
        "game_id": RECORDED,
        "gameday": "2003-10-27",
        "feed_gametime": "09:00",
        "corrected_gametime": "21:00",
        "source_url": "http://web.archive.org/web/1/https://example.org/box",
        "source_start_time": "9:09 PM ET",
    } | overrides
    return "[[correction]]\n" + "".join(f'{k} = "{v}"\n' for k, v in fields.items())


class TestTheLoaderRefusesByName:
    def test_a_well_formed_planted_record_loads(self, tmp_path: Path) -> None:
        loaded = ingest_games.load_kickoff_hour_corrections(_write(tmp_path, _entry()))
        assert loaded[RECORDED].corrected_gametime == "21:00"

    def test_an_empty_source_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(
            ingest_games.KickoffCorrectionRecordError, match="source_url"
        ):
            ingest_games.load_kickoff_hour_corrections(
                _write(tmp_path, _entry(source_url=""))
            )

    def test_a_duplicate_game_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(
            ingest_games.KickoffCorrectionRecordError, match="more than one"
        ):
            ingest_games.load_kickoff_hour_corrections(
                _write(tmp_path, _entry() + "\n" + _entry())
            )

    def test_a_cited_start_outside_the_corrected_hour_is_refused(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ingest_games.KickoffCorrectionRecordError, match="contradicts"
        ):
            ingest_games.load_kickoff_hour_corrections(
                _write(tmp_path, _entry(source_start_time="8:05 PM ET"))
            )

    def test_a_malformed_clock_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ingest_games.KickoffCorrectionRecordError, match="HH:MM"):
            ingest_games.load_kickoff_hour_corrections(
                _write(tmp_path, _entry(corrected_gametime="9pm"))
            )

    def test_a_correction_that_moves_nothing_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(
            ingest_games.KickoffCorrectionRecordError, match="to itself"
        ):
            ingest_games.load_kickoff_hour_corrections(
                _write(tmp_path, _entry(corrected_gametime="09:00"))
            )
