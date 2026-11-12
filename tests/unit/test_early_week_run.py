"""The Tuesday early-week run and its rehearsal (``pipeline/early_week.py``).

WHAT IS PROVEN
--------------
The run decides one coming NFL week or skips it by name, and writes nothing when it skips. The
heavy steps -- the silver refresh, the gold build, the early inputs, the scoring and the ledger
write -- are replaced through the module's seams, so these tests run in moments and never touch
the live lake. Everything else is the real code: the calendar, the board file reader, the odds
transform, the pricing and the decision.

Two tests go through the real ledger write as well, into a temporary ledger, because a board that
matched no games reaches it as an empty frame and the reproduction key has to accept that.

Every store lives under pytest's temporary directories; production ``ledger/``, ``artifacts/``,
``data/`` and ``outputs/`` are never read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from backtest.early_arm_2026 import EARLY_RECIPE, RECORD_PATH
from backtest.study_1b_inputs import EarlyInputs
from data.upstream_live import AS_OF_ENV, LIVE_MANIFEST_SCHEMA_VERSION
from features import elo_features
from features.elo_features import ProvisionalSnapshotAsTrainingInputError
from forward_ledger.canonical import ENTRY_KIND_ROW, GENESIS_HASH
from forward_ledger.runner import RecordOutcome, record_early_week
from forward_ledger.store import (
    _write_entries_unguarded,
    build_entry,
    ledger_path,
    read_entries,
)
from pipeline import early_week
from scripts.ingest_odds import OddsDataIngester
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_owner_install_records import V2_IDS, ledger_with_owner_install

TUESDAY = date(2026, 11, 17)
BOARD_AT = datetime(2026, 11, 17, 21, 40, tzinfo=UTC)  # 16:40 ET on the Tuesday
NOW_INSTANT = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
# The Thursday game's lock is Wednesday 18:00 ET; the Sunday game's is Saturday 18:00 ET.
THURSDAY_LOCK = datetime(2026, 11, 18, 18, 0, tzinfo=lock_rule.ET)
SUNDAY_LOCK = datetime(2026, 11, 21, 18, 0, tzinfo=lock_rule.ET)

MODEL_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

INGESTER = OddsDataIngester.__new__(OddsDataIngester)
INGESTER.sportsbook_priority = []


def _game(
    game_id: str,
    week: int,
    kickoff: str,
    *,
    home: str = "DET",
    away: str = "NE",
    roof: str = "outdoor",
    scored: bool = False,
) -> dict[str, Any]:
    """One silver ``games`` row, with the columns the run reads."""
    return {
        "game_id": game_id,
        "season": 2026,
        "week": week,
        "game_type": "REG",
        "home_team": home,
        "away_team": away,
        "kickoff_et": pd.Timestamp(kickoff),
        "venue_roof": roof,
        "neutral_site": False,
        "home_score": 24.0 if scored else float("nan"),
        "away_score": 21.0 if scored else float("nan"),
    }


# Week 10 ends on the Monday night before the Tuesday, so week 11 is due on 2026-11-17.
PLAYED = _game("g0", 10, "2026-11-17T01:15Z", scored=True)
THURSDAY_GAME = _game("g1", 11, "2026-11-20T01:15Z")
SUNDAY_GAME = _game("g2", 11, "2026-11-22T18:00Z")


def _board_line(captured_at: datetime, events: list[dict[str, Any]]) -> dict[str, Any]:
    """One line of ``early_lines.jsonl``, as the capture script writes it."""
    return {
        "captured_at_utc": captured_at.isoformat(),
        "remaining_credits_before": 400,
        "events": len(events),
        "board": events,
    }


@dataclass
class World:
    """The run's inputs, and what each seam was asked to do."""

    games: pd.DataFrame
    lines_path: Path
    gold_root: Path
    rehearsal_root: Path
    calls: list[str] = field(default_factory=list)
    refreshed: list[tuple[int, frozenset[str]]] = field(default_factory=list)
    build_locks: list[datetime] = field(default_factory=list)
    # The weather frame each gold build was handed: None means it reads silver.
    build_weather: list[pd.DataFrame | None] = field(default_factory=list)
    # Each in-memory weather computation the rehearsal asked for: (season, week's game ids).
    rehearsal_weather: list[tuple[int, frozenset[str]]] = field(default_factory=list)
    gold_dirs: list[Path] = field(default_factory=list)
    early_times: list[pd.Series] = field(default_factory=list)
    recorded: list[dict[str, Any]] = field(default_factory=list)
    predicted_games: list[str] = field(default_factory=list)
    # The games the early inputs leave out, as ``build_early_inputs`` does a neutral site.
    left_out: frozenset[str] = frozenset()
    predictions: dict[str, float] = field(
        default_factory=lambda: {"ats": 1.0, "ou": 45.0}
    )

    def write_board(self, captured_at: datetime, events: list[dict[str, Any]]) -> None:
        self.lines_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lines_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_board_line(captured_at, events)) + "\n")


#: What the rehearsal's in-memory weather computation returns in these tests.
REHEARSAL_WEATHER_FRAME = pd.DataFrame(
    {"game_id": ["g1", "g2"], "wind_mph": [5.0, 9.0]}
)


def _matrices(game_ids: list[str]) -> dict[str, pd.DataFrame]:
    return {
        target: pd.DataFrame(
            {"game_id": game_ids, "wind_mph": [float("nan")] * len(game_ids)}
        )
        for target in MODEL_TARGETS
    }


def _input_rows(game_ids: list[str]) -> dict[str, pd.DataFrame]:
    return {
        target: pd.DataFrame(
            {
                "game_id": game_ids,
                "season": [2026] * len(game_ids),
                "week": [11] * len(game_ids),
                "x": [1.0] * len(game_ids),
            }
        )
        for target in MODEL_TARGETS
    }


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """A week with two games, one board with nothing on it, and every heavy step replaced."""
    made = World(
        games=pd.DataFrame([PLAYED, THURSDAY_GAME, SUNDAY_GAME]),
        lines_path=tmp_path / "early_lines" / "early_lines.jsonl",
        gold_root=tmp_path / "early_gold",
        rehearsal_root=tmp_path / "early_gold_rehearsal",
    )
    made.write_board(BOARD_AT, [])

    def refresh(season: int, week_ids: frozenset[str]) -> None:
        made.calls.append("refresh")
        made.refreshed.append((season, week_ids))

    def build(
        unplayed_through_lock: datetime,
        *,
        weather_features: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        made.calls.append("build")
        made.build_locks.append(unplayed_through_lock)
        made.build_weather.append(weather_features)
        return _matrices([*made.games["game_id"].astype(str)])

    def rehearsal_weather(season: int, week_ids: frozenset[str]) -> pd.DataFrame:
        made.rehearsal_weather.append((season, week_ids))
        return REHEARSAL_WEATHER_FRAME

    def inputs(
        gold_dir: Path, games: pd.DataFrame, season: int, early_times: pd.Series
    ) -> EarlyInputs:
        made.calls.append("inputs")
        made.gold_dirs.append(gold_dir)
        made.early_times.append(early_times)
        kept = [str(g) for g in early_times.index if g not in made.left_out]
        return EarlyInputs(
            rows=_input_rows(kept),
            excluded=pd.DataFrame({"game_id": sorted(made.left_out)}),
            qb_values_clipped=0,
        )

    def score(
        rows: dict[str, pd.DataFrame], _resolved: Any
    ) -> dict[str, dict[str, float]]:
        made.calls.append("score")
        made.predicted_games.extend(str(g) for g in rows["ats"]["game_id"])
        return {
            target: dict.fromkeys(rows[target]["game_id"], made.predictions[target])
            for target in ("ats", "ou")
        }

    def record(bundle: Any, **options: Any) -> RecordOutcome:
        made.calls.append("record")
        made.recorded.append({"bundle": bundle, **options})
        return RecordOutcome(
            decided=len(bundle.frame),
            appended=len(bundle.frame),
            identical=0,
            refused=(),
            wrote=True,
            head_hash="h",
            entry_count=len(bundle.frame),
            appended_rows=tuple(bundle.frame.to_dict("records")),
        )

    monkeypatch.setattr(early_week, "REFRESH_SILVER", refresh)
    monkeypatch.setattr(early_week, "BUILD_GOLD", build)
    monkeypatch.setattr(early_week, "REHEARSAL_WEATHER", rehearsal_weather)
    monkeypatch.setattr(early_week, "BUILD_INPUTS", inputs)
    monkeypatch.setattr(early_week, "SCORE", score)
    monkeypatch.setattr(early_week, "RECORD", record)
    monkeypatch.setattr(early_week, "NOW", lambda: NOW_INSTANT)

    def games_table(_table: str, layer: str = "silver") -> pd.DataFrame:
        assert layer == "silver"
        return made.games

    monkeypatch.setattr(early_week, "load_dataframe", games_table)
    monkeypatch.setattr(early_week, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(early_week, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(early_week, "EARLY_LINES_FILE", made.lines_path)
    monkeypatch.setattr(early_week, "EARLY_GOLD_ROOT", made.gold_root)
    monkeypatch.setattr(early_week, "REHEARSAL_GOLD_ROOT", made.rehearsal_root)
    monkeypatch.setattr(early_week, "in_force_recipe", lambda *_: EARLY_RECIPE)
    monkeypatch.setattr(early_week, "resolve_production_artifacts", lambda *_: V2_IDS)
    monkeypatch.setattr(early_week, "odds_ingester", lambda: INGESTER)
    monkeypatch.setattr(
        elo_features,
        "load_elo_snapshots",
        lambda: pd.DataFrame({"game_id": ["g0"], "is_provisional": [False]}),
    )
    return made


def _ledger_entry(**overrides: Any) -> Any:
    """A stored early row of week 11, unless overridden."""
    fields: dict[str, Any] = {
        "game_id": "g1",
        "season": 2026,
        "week": 11,
        "arm": "early",
        **overrides,
    }
    return build_entry(GENESIS_HASH, 0, ENTRY_KIND_ROW, make_row(**fields))


def _store_ledger_rows(*entries: Any) -> None:
    _write_entries_unguarded(early_week.LEDGER_DIR, list(entries))


# ---------------------------------------------------------------------------
# The named skips: each writes nothing
# ---------------------------------------------------------------------------


def test_not_due_writes_nothing(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    result = early_week.run_early_week(date(2026, 11, 18), board_captured=True)

    assert result.outcome == early_week.OUTCOME_NOT_DUE
    assert (result.reason, result.season, result.week) == (None, None, None)
    assert (result.games, result.bets, result.declined) == (0, 0, 0)
    assert result.appended_rows == ()
    assert world.calls == []
    assert capsys.readouterr().out == (
        "EARLY_BETS= not_due season=None week=None games=0 bets=0 declined=0\n"
    )


def test_a_decided_week_is_not_decided_twice(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    _store_ledger_rows(_ledger_entry(status="suppressed"))

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.outcome == early_week.OUTCOME_SKIPPED
    assert result.reason == "already_decided"
    assert (result.season, result.week) == (2026, 11)
    assert world.calls == []
    assert capsys.readouterr().out == (
        "EARLY_BETS= skipped reason=already_decided season=2026 week=11 "
        "games=0 bets=0 declined=0\n"
    )


@pytest.mark.parametrize(
    "other",
    [{"arm": "live"}, {"week": 10}, {"season": 2025}],
    ids=["another_arm", "another_week", "another_season"],
)
def test_only_an_early_row_of_the_same_week_counts_as_decided(
    world: World, other: dict[str, Any]
) -> None:
    _store_ledger_rows(_ledger_entry(**other))

    assert (
        early_week.run_early_week(TUESDAY, board_captured=True).outcome
        == early_week.OUTCOME_DECIDED
    )


def test_an_unscored_earlier_game_skips_the_week(world: World) -> None:
    world.games.loc[world.games["game_id"] == "g0", "home_score"] = float("nan")

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.outcome == early_week.OUTCOME_SKIPPED
    assert result.reason == "earlier_week_unscored"
    assert world.recorded == [] and world.calls == []


def test_a_game_of_the_coming_week_needs_no_score(world: World) -> None:
    """Only a LOWER week's score is asked for; the week being decided has none yet."""
    result = early_week.run_early_week(TUESDAY, board_captured=True)
    assert result.outcome == early_week.OUTCOME_DECIDED


def test_no_board_skips_the_week(world: World) -> None:
    """The capture reported no board, whatever the file holds."""
    result = early_week.run_early_week(TUESDAY, board_captured=False)

    assert result.outcome == early_week.OUTCOME_SKIPPED
    assert result.reason == "no_board"
    assert world.calls == []


def test_a_board_from_another_day_is_no_board(world: World) -> None:
    world.lines_path.unlink()
    world.write_board(BOARD_AT - timedelta(days=1), [])

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.reason == "no_board"
    assert world.calls == []


def test_todays_board_torn_mid_write_is_no_board(world: World) -> None:
    """One torn line never raises: with no other board today, the week skips as no_board."""
    world.lines_path.write_text(
        '{"captured_at_utc": "2026-11-17T21:40:00+00:00", "board": [{"id"',
        encoding="utf-8",
    )

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.outcome == early_week.OUTCOME_SKIPPED
    assert result.reason == "no_board"
    assert world.calls == []


def test_old_models_skip_the_week(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        early_week, "in_force_recipe", lambda *_: "recipe-2026-refit-v1"
    )

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.outcome == early_week.OUTCOME_SKIPPED
    assert result.reason == "recipe_not_v2"
    assert world.calls == []


def test_the_skips_are_tried_in_order(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every skip holds at once; clearing each in turn reveals the next one, in the stated order."""
    _store_ledger_rows(_ledger_entry())
    world.games.loc[world.games["game_id"] == "g0", "home_score"] = float("nan")
    monkeypatch.setattr(early_week, "in_force_recipe", lambda *_: "another-recipe")

    def reason(*, board_captured: bool) -> str | None:
        return early_week.run_early_week(TUESDAY, board_captured=board_captured).reason

    assert reason(board_captured=False) == "already_decided"

    ledger_path(early_week.LEDGER_DIR).unlink()
    assert reason(board_captured=False) == "earlier_week_unscored"

    world.games.loc[world.games["game_id"] == "g0", "home_score"] = 24.0
    assert reason(board_captured=False) == "no_board"

    assert reason(board_captured=True) == "recipe_not_v2"
    assert world.calls == []


def test_a_failure_in_a_later_step_propagates(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daily run's wrapper names the failure; this module never swallows it."""

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("the gold build fell over")

    monkeypatch.setattr(early_week, "BUILD_GOLD", fail)

    with pytest.raises(RuntimeError, match="the gold build fell over"):
        early_week.run_early_week(TUESDAY, board_captured=True)
    assert world.recorded == []


# ---------------------------------------------------------------------------
# The silver refresh, and the check on the played games
# ---------------------------------------------------------------------------


def test_the_silver_refresh_is_for_the_coming_weeks_games(world: World) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)

    assert world.refreshed == [(2026, frozenset({"g1", "g2"}))]


def test_a_played_game_with_a_provisional_elo_row_stops_the_week(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        elo_features,
        "load_elo_snapshots",
        lambda: pd.DataFrame(
            {
                "game_id": ["g0"],
                "season": [2026],
                "week": [10],
                "is_provisional": [True],
            }
        ),
    )

    with pytest.raises(ProvisionalSnapshotAsTrainingInputError, match="g0"):
        early_week.run_early_week(TUESDAY, board_captured=True)

    assert world.calls == ["refresh"]
    assert world.recorded == []


def test_the_coming_weeks_own_provisional_rows_do_not_stop_it(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh writes a provisional row for every game of the coming week, by design."""
    monkeypatch.setattr(
        elo_features,
        "load_elo_snapshots",
        lambda: pd.DataFrame(
            {"game_id": ["g0", "g1"], "is_provisional": [False, True]}
        ),
    )

    assert (
        early_week.run_early_week(TUESDAY, board_captured=True).outcome
        == early_week.OUTCOME_DECIDED
    )


# ---------------------------------------------------------------------------
# The gold written for the week
# ---------------------------------------------------------------------------


def test_an_indoor_game_keeps_zero_wind(tmp_path: Path) -> None:
    matrices = {
        target: pd.DataFrame(
            {
                "game_id": ["indoor", "outdoor", "retractable", "older_indoor"],
                "wind_mph": [float("nan"), float("nan"), float("nan"), 7.0],
            }
        )
        for target in MODEL_TARGETS
    }
    week_games = pd.DataFrame(
        {
            "game_id": ["indoor", "outdoor", "retractable"],
            "venue_roof": ["indoor", "outdoor", "retractable"],
        }
    )

    assert early_week.write_week_gold(matrices, week_games, tmp_path) == tmp_path

    for target in MODEL_TARGETS:
        written = pd.read_parquet(tmp_path / f"features_{target}.parquet").set_index(
            "game_id"
        )
        assert written.loc["indoor", "wind_mph"] == 0.0
        assert pd.isna(written.loc["outdoor", "wind_mph"])
        assert pd.isna(written.loc["retractable", "wind_mph"])
        assert written.loc["older_indoor", "wind_mph"] == 7.0


def test_the_week_gold_is_never_written_under_data(tmp_path: Path) -> None:
    week_games = pd.DataFrame({"game_id": ["g1"], "venue_roof": ["outdoor"]})

    with pytest.raises(ValueError, match="data"):
        early_week.write_week_gold(_matrices(["g1"]), week_games, Path("data/gold"))


def test_the_gold_is_built_through_the_weeks_latest_lock(world: World) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)

    assert world.build_locks == [SUNDAY_LOCK]


def test_the_gold_goes_to_the_weeks_own_directory(world: World) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)

    expected = world.gold_root / "2026_w11"
    assert world.gold_dirs == [expected.resolve()]
    assert world.recorded[0]["gold_dir"] == expected.resolve()
    for target in MODEL_TARGETS:
        assert (expected / f"features_{target}.parquet").is_file()


# ---------------------------------------------------------------------------
# Games past their lock, and games with no prediction
# ---------------------------------------------------------------------------


def test_a_game_past_its_lock_is_left_out_and_the_rest_decided(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lock exactly at the clock has passed: only games strictly before it are decided."""
    monkeypatch.setattr(early_week, "NOW", lambda: THURSDAY_LOCK.astimezone(UTC))

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert "EARLY_LOCK_PASSED g1" in capsys.readouterr().out
    assert result.games == 1
    assert list(world.early_times[0].index) == ["g2"]
    bundle = world.recorded[0]["bundle"]
    assert list(bundle.schedule["game_id"]) == ["g2"]
    assert set(bundle.frame["game_id"]) == {"g2"}
    assert world.recorded[0]["publish_by"] == SUNDAY_LOCK


def test_a_game_a_second_before_its_lock_is_kept(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        early_week,
        "NOW",
        lambda: (THURSDAY_LOCK - timedelta(seconds=1)).astimezone(UTC),
    )

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.games == 2
    assert world.recorded[0]["publish_by"] == THURSDAY_LOCK


def test_every_kept_game_is_timed_at_the_boards_capture(world: World) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)

    times = world.early_times[0]
    assert sorted(times.index) == ["g1", "g2"]
    assert set(times) == {pd.Timestamp(BOARD_AT)}


def test_a_neutral_site_game_is_declined_for_no_prediction(world: World) -> None:
    """The early inputs leave the game out, so it reaches the decision with no number."""
    world.left_out = frozenset({"g1"})

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.games == 2 and result.bets == 0 and result.declined == 4
    assert set(world.predicted_games) == {"g2"}
    frame = world.recorded[0]["bundle"].frame
    g1 = frame.loc[frame["game_id"] == "g1"]
    assert set(g1["rejection_reason"]) == {"missing_prediction"}
    assert set(g1["status"]) == {"suppressed"}


def test_a_week_with_no_match_on_the_board_declines_every_row(world: World) -> None:
    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.bets == 0 and result.declined == 4
    frame = world.recorded[0]["bundle"].frame
    assert set(frame["rejection_reason"]) == {"missing_snapshot"}


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_steps_run_in_order(world: World) -> None:
    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert world.calls == ["refresh", "build", "inputs", "score", "record"]
    assert result.outcome == early_week.OUTCOME_DECIDED
    assert (result.season, result.week) == (2026, 11)
    assert result.reason is None


def test_the_record_gets_the_record_files_sha(world: World) -> None:
    """RE-POINTED 2026-11-09 for backtest/early_arm_2026.py: the hash is of the text as git holds it.

    The bytes on disk are CRLF in a core.autocrlf checkout, so they are normalized to LF first.
    """
    early_week.run_early_week(TUESDAY, board_captured=True)

    expected = hashlib.sha256(
        Path(RECORD_PATH).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert world.recorded[0]["meta_extra"] == {"early_arm_record_sha256": expected}


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_the_record_hash_does_not_depend_on_line_endings(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newline: str
) -> None:
    """core.autocrlf is true, so a checkout holds CRLF while git holds LF.

    Hashing the bytes as they sit on disk would make a later comparison with git show an edit
    that nobody made.
    """
    text = "first line\nsecond line\n"
    record_copy = tmp_path / "record_copy.py"
    record_copy.write_bytes(text.replace("\n", newline).encode("ascii"))
    monkeypatch.setattr(early_week, "RECORD_PATH", str(record_copy))

    early_week.run_early_week(TUESDAY, board_captured=True)

    expected = hashlib.sha256(text.encode("ascii")).hexdigest()
    assert world.recorded[0]["meta_extra"] == {"early_arm_record_sha256": expected}


def test_the_record_receives_the_decision_and_its_inputs(world: World) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)

    options = world.recorded[0]
    bundle = options["bundle"]
    assert options["decided_at"] == NOW_INSTANT
    assert options["publish_by"] == THURSDAY_LOCK
    assert options["ledger_dir"] == early_week.LEDGER_DIR
    assert options["artifacts_dir"] == early_week.ARTIFACTS_DIR
    assert options["repo_dir"] == Path()
    assert bundle.resolved == V2_IDS
    assert bundle.fits == {}
    assert bundle.chain_fit_path == Path("config/key_number_tables.json")
    assert bundle.run_instant == NOW_INSTANT
    assert list(bundle.schedule.columns) == [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "kickoff_et",
    ]
    assert sorted(bundle.gold_inputs) == sorted(MODEL_TARGETS)


def test_the_result_counts_the_week_and_carries_the_new_rows(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert (result.games, result.bets, result.declined) == (2, 0, 4)
    assert len(result.appended_rows) == 4
    assert capsys.readouterr().out.endswith(
        "EARLY_BETS= decided season=2026 week=11 games=2 bets=0 declined=4\n"
    )


# ---------------------------------------------------------------------------
# The real board, the real ledger write
# ---------------------------------------------------------------------------

_MARKET_STAMP = "2026-11-17T21:20:00Z"


def _spreads(price_home: int) -> dict[str, Any]:
    return {
        "key": "spreads",
        "last_update": _MARKET_STAMP,
        "outcomes": [
            {"name": "Detroit Lions", "price": price_home, "point": -3.5},
            {"name": "New England Patriots", "price": -110, "point": 3.5},
        ],
    }


def _totals() -> dict[str, Any]:
    return {
        "key": "totals",
        "last_update": _MARKET_STAMP,
        "outcomes": [
            {"name": "Over", "price": -110, "point": 47.5},
            {"name": "Under", "price": -110, "point": 47.5},
        ],
    }


def _book(key: str, price_home: int = -110) -> dict[str, Any]:
    return {
        "key": key,
        "last_update": _MARKET_STAMP,
        "markets": [_spreads(price_home), _totals()],
    }


NE_AT_DET = "2026_W11_NE@DET"
NE_AT_DET_EVENT: dict[str, Any] = {
    "id": "abc123",
    "commence_time": "2026-11-22T18:00:00Z",
    "home_team": "Detroit Lions",
    "away_team": "New England Patriots",
    "bookmakers": [_book("draftkings"), _book("fanduel", -105), _book("betmgm")],
}


@pytest.fixture
def real_ledger(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The real ledger write into a temporary ledger holding the installed v2 generation."""
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    ledger = ledger_with_owner_install(tmp_path, installed=True)
    artifacts = tmp_path / "artifacts"
    for artifact_id in (
        V2_IDS.wp,
        V2_IDS.ats,
        V2_IDS.ou,
        V2_IDS.blend,
        V2_IDS.converter,
    ):
        directory = artifacts / str(artifact_id)
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    manifests = tmp_path / "upstream_live"
    manifests.mkdir()
    (manifests / "2026.json").write_text(
        json.dumps(
            {
                "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
                "season": 2026,
                "datasets": {
                    "schedules": {
                        "captures": [
                            {
                                "week": 11,
                                "sequence": 1,
                                "sha256": "s" * 64,
                                "path": "bronze/s",
                                "captured_at_utc": "2026-11-17T21:30:00+00:00",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def record(bundle: Any, **options: Any) -> RecordOutcome:
        world.calls.append("record")
        outcome = record_early_week(
            bundle,
            **{
                **options,
                "repo_dir": None,
                "manifest_dir": manifests,
                "log": lambda *_args, **_fields: None,
                # The deadline is judged on this clock, so the test holds on any real date.
                "clock": lambda: NOW_INSTANT,
            },
        )
        world.recorded.append({"bundle": bundle, **options})
        return outcome

    monkeypatch.setattr(early_week, "LEDGER_DIR", ledger)
    monkeypatch.setattr(early_week, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(early_week, "RECORD", record)
    return ledger


def test_a_board_matching_no_games_reaches_the_ledger_with_every_row_declined(
    world: World, real_ledger: Path
) -> None:
    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert result.outcome == early_week.OUTCOME_DECIDED
    assert result.bets == 0 and result.declined == 4
    assert len(result.appended_rows) == 4
    assert {row["rejection_reason"] for row in result.appended_rows} == {
        "missing_snapshot"
    }
    assert {entry.immutable["arm"] for entry in read_entries(real_ledger)} == {"early"}


def test_a_decided_week_is_not_decided_again_from_the_real_ledger(
    world: World, real_ledger: Path
) -> None:
    early_week.run_early_week(TUESDAY, board_captured=True)
    world.calls.clear()

    second = early_week.run_early_week(TUESDAY, board_captured=True)

    assert second.reason == "already_decided"
    assert world.calls == []


def test_a_priced_game_goes_through_to_a_bet_on_the_ledger(
    world: World, real_ledger: Path
) -> None:
    """The real board, the real pricing and the real ledger write, end to end."""
    world.games = pd.DataFrame([PLAYED, _game(NE_AT_DET, 11, "2026-11-22T18:00Z")])
    world.lines_path.unlink()
    world.write_board(BOARD_AT, [NE_AT_DET_EVENT])
    # A spread far from the 3.5 line clears the floor; the total sits on the consensus.
    world.predictions = {"ats": 21.5, "ou": 47.5}

    result = early_week.run_early_week(TUESDAY, board_captured=True)

    assert (result.games, result.bets, result.declined) == (1, 1, 1)
    rows = {row["target"]: row for row in result.appended_rows}
    assert rows["ats"]["status"] == "live"
    assert rows["ats"]["eligibility_label"] == "book:fanduel"
    assert rows["ats"]["selected_odds"] == -105.0
    assert rows["ou"]["status"] == "suppressed"
    assert rows["ou"]["rejection_reason"] == "no_bet_side"
    assert {entry.immutable["recipe_id"] for entry in read_entries(real_ledger)} == {
        EARLY_RECIPE
    }


# ---------------------------------------------------------------------------
# The rehearsal
# ---------------------------------------------------------------------------


def test_the_rehearsal_writes_no_ledger_row(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rehearsal uses the latest board of any day, and only touches its own directory."""
    world.lines_path.unlink()
    world.write_board(BOARD_AT - timedelta(days=1), [])

    early_week.rehearse_early_week(11)

    assert "record" not in world.calls and "refresh" not in world.calls
    assert world.calls == ["build", "inputs", "score"]
    assert not (world.gold_root).exists()
    assert (world.rehearsal_root / "features_ou.parquet").is_file()
    lines = [
        line for line in capsys.readouterr().out.splitlines() if "consensus=" in line
    ]
    assert len(lines) == 4
    assert lines[0] == (
        "g1 ats consensus=- model=1.00 side=- book=- price=- ev=- status=missing_snapshot"
    )


def test_the_rehearsal_shows_a_priced_bet(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    world.games = pd.DataFrame([PLAYED, _game(NE_AT_DET, 11, "2026-11-22T18:00Z")])
    world.lines_path.unlink()
    world.write_board(BOARD_AT, [NE_AT_DET_EVENT])
    world.predictions = {"ats": 21.5, "ou": 47.5}

    early_week.rehearse_early_week(11)

    out = capsys.readouterr().out
    ats = next(line for line in out.splitlines() if line.startswith(f"{NE_AT_DET} ats"))
    assert "consensus=3.50" in ats and "model=21.50" in ats
    assert "side=home_cover" in ats and "book=fanduel" in ats
    assert "price=-105" in ats and "status=live" in ats
    assert "ev=-" not in ats


def test_the_rehearsal_builds_the_week_with_its_weather_computed_in_memory(
    world: World,
) -> None:
    """Fixed 2026-11-11 after the first real rehearsal crashed at the weather gate.

    The rehearsal skips the silver refresh, and silver weather_features holds only games whose
    slate night has run, so the week's other games reached the gold build with no weather row.
    The rehearsal now computes the week's weather as the refresh would, in memory, and hands it to
    the build; nothing is written to silver.
    """
    early_week.rehearse_early_week(11)

    assert world.rehearsal_weather == [(2026, frozenset({"g1", "g2"}))]
    assert len(world.build_weather) == 1
    assert world.build_weather[0] is REHEARSAL_WEATHER_FRAME
    assert "refresh" not in world.calls


def test_the_run_builds_its_gold_from_silver_weather(world: World) -> None:
    """The real run refreshes silver first, so its build reads silver and computes nothing aside."""
    early_week.run_early_week(TUESDAY, board_captured=True)

    assert world.build_weather == [None]
    assert world.rehearsal_weather == []


def test_the_rehearsal_builder_reads_the_weather_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v2 builder reads silver weather_features from parquet; the rehearsal's reads its frame."""
    from scripts import build_features

    reads: list[tuple[Any, ...]] = []
    silver = pd.DataFrame({"game_id": ["g1"], "wind_mph": [1.0]})

    def load(*args: Any) -> pd.DataFrame:
        reads.append(args)
        return silver

    monkeypatch.setattr(build_features, "load_dataframe", load)
    given = pd.DataFrame({"game_id": ["g1", "g2"], "wind_mph": [5.0, 9.0]})

    assert (
        early_week.FeatureMatrixBuilderV2Live()._read_silver_weather_features()
        is silver
    )
    assert reads == [("weather_features", "silver", "parquet")]

    read = early_week.GoldBuilderWithWeather(given)._read_silver_weather_features()
    pd.testing.assert_frame_equal(read, given)
    assert read is not given
    assert len(reads) == 1


def test_a_rehearsal_with_no_board_is_refused_by_name(world: World) -> None:
    world.lines_path.unlink()

    with pytest.raises(RuntimeError, match="board"):
        early_week.rehearse_early_week(11)


def test_the_command_line_rehearses_the_week_named(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    rehearsed: list[int] = []
    monkeypatch.setattr(early_week, "rehearse_early_week", rehearsed.append)

    assert early_week.main(["--rehearse-week", "11"]) == 0
    assert rehearsed == [11]
