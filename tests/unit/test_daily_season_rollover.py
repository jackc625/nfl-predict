"""The daily run rolls into the next season with no manual step (step 27b of Plan 33.2-27).

Before this, the run picked its season from the CALENDAR and the live capture owned exactly one
season (2026), so on 2027-08-01 the schedule resolver refused ("the 2027 schedule is not
recorded"), the capture refused 2027 ("no zone owns it"), and nothing could record 2027 until a
human edited the zone boundary. Now:

* the season comes from the RECORDED schedule (``utils.current_slate.refresh_target``): the
  current slate's season, or -- once the calendar season has turned past a completed season --
  the next season's week 1, whose capture is how its schedule gets recorded;
* the live capture opens a season after the first live one only once the recorded schedule holds
  the previous season's Super Bowl, so it can never open a season early or two ahead;
* before any game of that season has kicked off there are no plays: play-by-play is not captured
  and no results are ingested, each skipped by name;
* if nflverse genuinely serves no schedule for the due season, the run refuses by name.

Every run date below is FROZEN (the run resolves at its date's 18:00 ET lock, not the wall clock)
and every schedule is a fixture. Nothing touches the network or a production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

import scripts.daily_lock_pipeline as daily
from data import upstream_pin
from data.upstream_pin import LIVE_ZONE_FIRST_SEASON, ZoneWriteRefused
from pipeline.daily_steps import slate_lock
from utils import current_slate
from utils.date_utils import ET

FIRST = LIVE_ZONE_FIRST_SEASON  # 2026, the first live season
NEXT = FIRST + 1

# The first live season's tail, Super Bowl included, and the next season's opening week.
FIRST_SEASON_TAIL = (
    (f"{FIRST}_W18_A@B", FIRST, 18, f"{NEXT}-01-10 13:00", "REG"),
    (f"{FIRST}_W19_C@D", FIRST, 19, f"{NEXT}-01-16 16:30", "WC"),
    (f"{FIRST}_W20_E@F", FIRST, 20, f"{NEXT}-01-23 16:30", "DIV"),
    (f"{FIRST}_W21_G@H", FIRST, 21, f"{NEXT}-01-31 15:00", "CON"),
    (f"{FIRST}_W22_I@J", FIRST, 22, f"{NEXT}-02-14 18:30", "SB"),
)
NEXT_SEASON_HEAD = (
    (f"{NEXT}_W01_K@L", NEXT, 1, f"{NEXT}-09-09 20:20", "REG"),
    (f"{NEXT}_W01_M@N", NEXT, 1, f"{NEXT}-09-12 13:00", "REG"),
    (f"{NEXT}_W02_O@P", NEXT, 2, f"{NEXT}-09-16 20:15", "REG"),
)


def _schedule(*rows: tuple[str, int, int, str, str]) -> pd.DataFrame:
    """A silver ``games``-shaped fixture: (game_id, season, week, 'YYYY-MM-DD HH:MM' ET, type)."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "kickoff_et": pd.Timestamp(kickoff, tz=ET),
                "game_type": game_type,
                "home_team": game_id.split("@")[1],
                "away_team": game_id.split("_")[2].split("@")[0],
            }
            for game_id, season, week, kickoff, game_type in rows
        ]
    )


def _record_schedule(monkeypatch, schedule: pd.DataFrame) -> None:
    """Make *schedule* the recorded silver ``games`` every resolver reads."""
    prepared = current_slate.prepare_schedule(schedule)
    monkeypatch.setattr(current_slate, "load_recorded_schedule", lambda *_a: prepared)


def _et(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=ET)


# ---------------------------------------------------------------------------
# Which season the run refreshes
# ---------------------------------------------------------------------------


class TestTheSeasonComesFromTheSchedule:
    def test_august_after_a_completed_season_refreshes_the_next_one(self) -> None:
        frame = _schedule(*FIRST_SEASON_TAIL)
        assert current_slate.refresh_target(
            _et(f"{NEXT}-08-02 18:00"), schedule=frame
        ) == (NEXT, 1)

    def test_the_spring_still_refreshes_the_completed_season(self) -> None:
        frame = _schedule(*FIRST_SEASON_TAIL)
        season, _week = current_slate.refresh_target(
            _et(f"{NEXT}-03-15 18:00"), schedule=frame
        )
        assert season == FIRST

    def test_once_recorded_the_new_season_opens_on_its_openers_lock_day(self) -> None:
        frame = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        before = current_slate.refresh_target(
            _et(f"{NEXT}-08-20 18:00"), schedule=frame
        )
        lock_day = current_slate.refresh_target(
            _et(f"{NEXT}-09-08 18:00"), schedule=frame
        )
        assert before[0] == FIRST  # the documented offseason value, not a guess
        assert lock_day == (NEXT, 1)

    def test_every_other_refusal_still_propagates(self) -> None:
        """A store stale mid-season is not a season roll: it stays a named refusal."""
        truncated = _schedule(FIRST_SEASON_TAIL[0])
        with pytest.raises(current_slate.ScheduleIncompleteError):
            current_slate.refresh_target(_et(f"{NEXT}-01-30 18:00"), schedule=truncated)

    def test_kicked_off_is_read_from_the_recorded_kickoffs(self) -> None:
        frame = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        opener = _et(f"{NEXT}-09-09 20:20")
        assert not current_slate.season_has_kicked_off(
            NEXT, opener - timedelta(seconds=1), schedule=frame
        )
        assert current_slate.season_has_kicked_off(NEXT, opener, schedule=frame)
        assert not current_slate.season_has_kicked_off(NEXT + 1, opener, schedule=frame)


# ---------------------------------------------------------------------------
# The live capture opens the next season only after the last one ended
# ---------------------------------------------------------------------------


class TestTheLiveCaptureOpensTheNextSeasonOnlyAfterTheLastEnds:
    def test_the_next_season_is_refused_while_the_last_has_no_super_bowl(
        self, monkeypatch
    ) -> None:
        from scripts.capture_live_season import refuse_an_unopened_live_season

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL[:-1]))
        with pytest.raises(
            ZoneWriteRefused, match=f"no Super Bowl\\s+for season {FIRST}"
        ):
            refuse_an_unopened_live_season(NEXT)

    def test_the_next_season_opens_once_the_super_bowl_is_recorded(
        self, monkeypatch
    ) -> None:
        from scripts.capture_live_season import refuse_an_unopened_live_season

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL))
        refuse_an_unopened_live_season(NEXT)  # no refusal
        # The first live season needs no predecessor.
        refuse_an_unopened_live_season(FIRST)

    def test_a_season_two_ahead_is_refused(self, monkeypatch) -> None:
        from scripts.capture_live_season import refuse_an_unopened_live_season

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL))
        with pytest.raises(ZoneWriteRefused, match=f"season {NEXT}"):
            refuse_an_unopened_live_season(NEXT + 1)

    def test_the_capture_refuses_before_any_fetch(self, monkeypatch, tmp_path) -> None:
        import scripts.capture_live_season as capture
        from scripts import pin_upstream_snapshot

        def _explode(*_a, **_k):
            raise AssertionError("an unopened season reached the fetch")

        monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _explode)
        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL[:-1]))

        with pytest.raises(ZoneWriteRefused):
            capture.capture_live_dataset(
                "schedules",
                NEXT,
                1,
                data_root=tmp_path / "data",
                manifest_dir=tmp_path / "upstream_live",
            )
        assert not (tmp_path / "upstream_live").exists()


# ---------------------------------------------------------------------------
# The daily run on a frozen date
# ---------------------------------------------------------------------------


@pytest.fixture
def refresh_env(monkeypatch, tmp_path):
    """Stub the capture and the ingest; record what the run asked of them."""
    from pipeline import steps
    from scripts import ingest_games

    asked: dict[str, object] = {}

    def _capture(*, season, week, datasets):
        asked["capture"] = (season, week, tuple(datasets))

    class _Ingester:
        def ingest_games(self, *, seasons, include_results):
            asked["ingest"] = (tuple(seasons), include_results)

    monkeypatch.setattr(steps, "step_capture_live_season", _capture)
    monkeypatch.setattr(ingest_games, "GameDataIngester", _Ingester)
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", tmp_path / "daily.jsonl")
    return asked


def _served(monkeypatch, frame: pd.DataFrame) -> None:
    """What the capture recorded nflverse serving, as the pinned loader reads it."""
    monkeypatch.setattr(upstream_pin, "load_schedules", lambda seasons, **_k: frame)


class TestTheDailyRunRollsIntoTheNextSeason:
    def test_an_august_day_captures_the_next_season_and_is_a_clean_no_op(
        self, monkeypatch, refresh_env, tmp_path, capsys
    ) -> None:
        run_date = date(NEXT, 8, 2)
        recorded = _schedule(*FIRST_SEASON_TAIL)
        _record_schedule(monkeypatch, recorded)
        _served(monkeypatch, _schedule(*NEXT_SEASON_HEAD))
        after_ingest = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        monkeypatch.setattr(
            "data.storage.load_dataframe", lambda *_a, **_k: after_ingest.copy()
        )

        start = slate_lock(run_date).astimezone(UTC) - timedelta(hours=6)
        assert daily.run_daily(run_date, start=start, dry_run=False) == 0

        assert refresh_env["capture"] == (NEXT, 1, ("depth_charts", "schedules"))
        assert refresh_env["ingest"] == ((NEXT,), False)
        out = capsys.readouterr().out
        assert f"CAPTURE_SKIPPED pbp: no {NEXT} game has kicked off" in out
        assert "NEXT_DAY_GAMES= 0" in out
        record = json.loads((tmp_path / "daily.jsonl").read_text(encoding="utf-8"))
        assert record["outcome"] == "no_games"

    def test_an_unpublished_next_schedule_is_refused_by_name(
        self, monkeypatch, refresh_env
    ) -> None:
        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL))
        _served(monkeypatch, _schedule(*NEXT_SEASON_HEAD).iloc[0:0])

        with pytest.raises(
            daily.ScheduleNotPublishedError, match=f"no {NEXT} schedule"
        ):
            daily._refresh_schedule(date(NEXT, 8, 2), dry_run=False)
        assert "ingest" not in refresh_env

    def test_in_season_every_dataset_and_the_results_are_refreshed(
        self, monkeypatch, refresh_env
    ) -> None:
        schedule = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        _record_schedule(monkeypatch, schedule)
        _served(monkeypatch, schedule)

        season = daily._refresh_schedule(date(NEXT, 9, 11), dry_run=False)

        assert season == NEXT
        assert refresh_env["capture"] == (
            NEXT,
            1,
            ("depth_charts", "pbp", "schedules"),
        )
        assert refresh_env["ingest"] == ((NEXT,), True)
