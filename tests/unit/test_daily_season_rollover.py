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

    def test_the_spring_refreshes_nothing(self) -> None:
        """33.2 review B WR-03 = C1 WR-03. Was: the spring refreshed the COMPLETED season.

        That appended a git-tracked capture every day for a season that cannot change, and
        failed every day once the season was sealed.
        """
        frame = _schedule(*FIRST_SEASON_TAIL)
        assert (
            current_slate.refresh_target(_et(f"{NEXT}-03-15 18:00"), schedule=frame)
            is None
        )

    def test_once_recorded_the_new_season_is_refreshed_until_its_opener(self) -> None:
        """Was: August refreshed the completed season as "week 18" and never the new one.

        A moved opener would then be missed, because the opener's lock day is read from the
        new season's record.
        """
        frame = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        before = current_slate.refresh_target(
            _et(f"{NEXT}-08-20 18:00"), schedule=frame
        )
        lock_day = current_slate.refresh_target(
            _et(f"{NEXT}-09-08 18:00"), schedule=frame
        )
        assert before == (NEXT, 1)
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
        # Was (step 27b): "no {NEXT} game has kicked off". Step 27c records an empty capture
        # once the schedule proves that; before it is recorded, nothing proves it.
        assert f"CAPTURE_SKIPPED pbp: the {NEXT} schedule is not recorded yet" in out
        assert "NEXT_DAY_GAMES= 0" in out
        record = json.loads((tmp_path / "daily.jsonl").read_text(encoding="utf-8"))
        assert record["outcome"] == "no_games"

    def test_a_spring_day_is_a_clean_no_op_with_no_capture_and_no_ingest(
        self, monkeypatch, refresh_env, tmp_path, capsys
    ) -> None:
        """33.2 review B WR-03 = C1 WR-03: no capture entry, no probe line, no re-ingest."""
        recorded = _schedule(*FIRST_SEASON_TAIL)
        _record_schedule(monkeypatch, recorded)
        scored = recorded.assign(home_score=20.0, away_score=17.0)
        monkeypatch.setattr(
            "data.storage.load_dataframe", lambda *_a, **_k: scored.copy()
        )

        run_date = date(NEXT, 4, 15)
        start = slate_lock(run_date).astimezone(UTC) - timedelta(hours=1)
        assert daily.run_daily(run_date, start=start, dry_run=False) == 0

        assert "capture" not in refresh_env
        assert "ingest" not in refresh_env
        out = capsys.readouterr().out
        assert "REFRESH_SKIPPED offseason" in out
        record = json.loads((tmp_path / "daily.jsonl").read_text(encoding="utf-8"))
        assert record["outcome"] == "offseason"

    def test_the_offseason_refreshes_the_completed_season_until_its_final_result_lands(
        self, monkeypatch, refresh_env
    ) -> None:
        recorded = _schedule(*FIRST_SEASON_TAIL)
        _record_schedule(monkeypatch, recorded)
        _served(monkeypatch, recorded)
        # The Super Bowl has been played; its result is not ingested yet.
        pending = recorded.assign(home_score=20.0, away_score=17.0)
        pending.loc[pending["game_type"] == "SB", ["home_score", "away_score"]] = None
        monkeypatch.setattr(
            "data.storage.load_dataframe", lambda *_a, **_k: pending.copy()
        )

        assert daily._refresh_schedule(date(NEXT, 2, 16), dry_run=False) == FIRST
        assert refresh_env["capture"][:2] == (FIRST, 22)
        assert refresh_env["ingest"] == ((FIRST,), True)

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


# ---------------------------------------------------------------------------
# The opener's lock day reads an empty season (step 27c, applying D32-03)
# ---------------------------------------------------------------------------

#: The 2027 opener is Thursday 2027-09-09 20:20 ET, so its lock day is Wednesday 2027-09-08.
OPENER_LOCK_DAY = date(NEXT, 9, 8)
#: The frozen capture instant: noon ET on the lock day, before the 18:00 lock.
LOCK_DAY_NOON = _et(f"{NEXT}-09-08 12:00")
#: The instant after the opener kicked off, for the outage case.
AFTER_OPENER = _et(f"{NEXT}-09-10 12:00")

#: nflreadpy's own season-window refusal, as ``load_pbp`` raises it before the Thursday flip.
WINDOW_REFUSAL = f"Season must be between 1999 and {FIRST}"

_PBP_TEXT_COLUMNS = frozenset(
    {"game_id", "posteam", "defteam", "home_team", "away_team", "play_type"}
)


def _prior_season_pbp() -> pd.DataFrame:
    """Two pass plays of the first live season, every pinned column, real-shaped dtypes."""
    int_values = {"season": FIRST, "week": 22, "home_score": 24, "away_score": 17}
    data: dict[str, object] = {}
    for column in upstream_pin.PBP_PINNED_COLUMNS:
        if column in int_values:
            data[column] = pd.array([int_values[column]] * 2, dtype="int32")
        elif column == "game_id":
            data[column] = [f"{FIRST}_W22_I@J"] * 2
        elif column in _PBP_TEXT_COLUMNS:
            data[column] = ["I", "J"]
        elif column == "passer_player_id":
            data[column] = ["00-0000001", "00-0000002"]
        else:
            data[column] = [0.25, -0.5]
    return pd.DataFrame(data)


@pytest.fixture
def live_dirs(tmp_path, monkeypatch, sealed_probe_offline):
    """A scratch data root and live manifest directory holding the first season's plays.

    The pinned reader's defaults point at them too, so a builder that calls it with no
    arguments reads the scratch zone. The sealed probe stays offline (``sealed_probe_offline``).
    """
    import scripts.capture_live_season as capture
    from data import upstream_live
    from scripts import pin_upstream_snapshot

    roots = {"data_root": tmp_path / "data", "manifest_dir": tmp_path / "upstream_live"}
    monkeypatch.setattr(upstream_pin, "default_data_root", lambda: roots["data_root"])
    monkeypatch.setattr(upstream_live, "LIVE_MANIFEST_DIR", roots["manifest_dir"])

    prior = _prior_season_pbp()
    monkeypatch.setattr(
        pin_upstream_snapshot, "fetch_live", lambda dataset, season: prior.copy()
    )
    capture.capture_live_dataset("pbp", FIRST, 22, **roots)
    return roots


def _refuse_next_season(monkeypatch) -> None:
    from scripts import pin_upstream_snapshot

    def _outside_window(dataset: str, season: int) -> pd.DataFrame:
        raise ValueError(WINDOW_REFUSAL)

    monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _outside_window)


def _freeze_capture_clock(monkeypatch, instant: datetime) -> None:
    import scripts.capture_live_season as capture

    monkeypatch.setattr(capture, "_capture_instant", lambda: instant.astimezone(UTC))


class TestTheOpenersLockDayReadsAnEmptySeason:
    def test_the_lock_day_run_captures_play_by_play_and_ingests_no_results(
        self, monkeypatch, refresh_env
    ) -> None:
        schedule = _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD)
        _record_schedule(monkeypatch, schedule)
        _served(monkeypatch, schedule)

        assert daily._refresh_schedule(OPENER_LOCK_DAY, dry_run=False) == NEXT
        assert refresh_env["capture"] == (
            NEXT,
            1,
            ("depth_charts", "pbp", "schedules"),
        )
        assert refresh_env["ingest"] == ((NEXT,), False)

    def test_the_refusal_is_recorded_as_an_empty_capture_and_the_build_reads_it(
        self, monkeypatch, live_dirs
    ) -> None:
        import scripts.capture_live_season as capture
        from features.qb_tracking import QBTracker

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD))
        _refuse_next_season(monkeypatch)
        _freeze_capture_clock(monkeypatch, LOCK_DAY_NOON)

        entry = capture.capture_live_dataset("pbp", NEXT, 1, **live_dirs)

        assert entry["rows"] == 0
        assert entry["upstream_width"] == 0
        assert entry["week_digests"] == {}
        assert len(entry["sha256"]) == 64
        assert WINDOW_REFUSAL in entry[capture.EMPTY_CAPTURE_BASIS_KEY]
        assert f"{NEXT}-09-09T20:20:00" in entry[capture.EMPTY_CAPTURE_BASIS_KEY]

        prior = upstream_pin.load_pbp([FIRST])
        empty = upstream_pin.load_pbp([NEXT])
        assert empty.empty
        assert list(empty.columns) == list(prior.columns)
        assert empty.dtypes.equals(prior.dtypes)

        both = upstream_pin.load_pbp([FIRST, NEXT])
        pd.testing.assert_frame_equal(both, prior)

        # The QB builder asks for the opener's season and the one before: it now reads the
        # prior season's plays instead of raising UpstreamPinMissing.
        plays = QBTracker()._load_pbp_data(NEXT)
        assert len(plays) == len(prior)
        assert plays["season"].dtype == prior["season"].dtype

    def test_a_played_game_with_nothing_upstream_still_refuses(
        self, monkeypatch, live_dirs
    ) -> None:
        import scripts.capture_live_season as capture

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD))
        _refuse_next_season(monkeypatch)
        _freeze_capture_clock(monkeypatch, AFTER_OPENER)

        with pytest.raises(upstream_pin.UpstreamSeasonWindowRefused):
            capture.capture_live_dataset("pbp", NEXT, 1, **live_dirs)
        manifest = json.loads(
            (live_dirs["manifest_dir"] / f"{FIRST}.json").read_text(encoding="utf-8")
        )
        assert len(manifest["datasets"]["pbp"]["captures"]) == 1
        assert not (live_dirs["manifest_dir"] / f"{NEXT}.json").exists()

    def test_an_unrecorded_schedule_proves_nothing_and_still_refuses(
        self, monkeypatch, live_dirs
    ) -> None:
        import scripts.capture_live_season as capture

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL))
        _refuse_next_season(monkeypatch)
        _freeze_capture_clock(monkeypatch, LOCK_DAY_NOON)

        with pytest.raises(upstream_pin.UpstreamSeasonWindowRefused):
            capture.capture_live_dataset("pbp", NEXT, 1, **live_dirs)
        assert not (live_dirs["manifest_dir"] / f"{NEXT}.json").exists()

    def test_only_play_by_play_is_empty_by_definition(
        self, monkeypatch, live_dirs
    ) -> None:
        import scripts.capture_live_season as capture

        _record_schedule(monkeypatch, _schedule(*FIRST_SEASON_TAIL, *NEXT_SEASON_HEAD))
        _refuse_next_season(monkeypatch)
        _freeze_capture_clock(monkeypatch, LOCK_DAY_NOON)

        with pytest.raises(upstream_pin.UpstreamSeasonWindowRefused):
            capture.capture_live_dataset("depth_charts", NEXT, 1, **live_dirs)
