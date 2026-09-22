"""The starting-QB identity and the QB play-by-play, both fenced at each game's lock.

Plan 33.2-13 Task 2 (SPEC R5 / R2). Two defects this module exists to keep dead:

* THE 2025 STARTER WAS HINDSIGHT. nflverse stopped assigning weeks to depth charts after
  2024 and appends every update with an ISO-8601 ``dt`` instead. The loader stamped every
  2025 row ``week = 0``, so no chart ever matched a target week and the resolution fell
  through to the most recent primary passer in play-by-play fenced by ``now`` -- the
  season's FINAL primary passer. Now a ``dt`` frame resolves QB1 from the latest snapshot
  published at or before the game's lock, and a ``week`` frame from that week's chart.
* THE PLAY-BY-PLAY FENCE WAS ONE FRAME-WIDE CUTOFF, applied as (season, week) pairs. A
  week with ONE finished game admitted every play of that week, the target game's own
  included. Now each PBP game is timed individually (kickoff plus the declared duration)
  and admitted only when it ended at or before the target game's lock.

The real-data cases read the PINNED upstream snapshots and silver ``games`` read-only.

INTERPRETATION RECORDED (the plan's wording): "no resolved starter is a passer whose first
start came after that week's lock". A legitimate starter's first start can be the target
game itself, which kicks off AFTER its own lock by construction, so a literal reading would
flag every week-1 starter. The hindsight the plan names is a passer who first started in a
LATER game; that is what is asserted: the resolved QB1's first 2025 start is this game or
an earlier one, never a later one.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import features.qb_tracking as qb_module
import utils.game_lock as lock_rule
from features.protocol import InformationTimeProvider
from features.provenance import (
    DECLARED_GAME_DURATION,
    PROVENANCE_COLUMNS,
    InformationBasis,
    InformationTimeGate,
    SourceCheckState,
)
from features.qb_tracking import QBTracker

ET = ZoneInfo("America/New_York")
_ONE_SECOND = timedelta(seconds=1)
_AS_OF_LATE = datetime(2030, 1, 1, tzinfo=ET)

# The 2025 weeks sampled from real data: spread across the season so a starter change,
# an injury and a benching all have a chance to appear.
SAMPLED_2025_WEEKS: tuple[int, ...] = (1, 4, 8, 12, 16, 18)


# ---------------------------------------------------------------------------
# Real 2025 data (pinned snapshots, read-only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_2025():
    games = pd.read_parquet("data/silver/games.parquet")
    tracker = QBTracker()
    depth = tracker._load_depth_charts(2025)
    pbp = tracker._load_pbp_data(2025)
    ends = tracker.pbp_game_end_times(pbp, games[games["season"].isin([2024, 2025])])

    stats = tracker.compute_per_game_qb_stats(pbp[pbp["season"] == 2025])
    stats["team"] = stats["posteam"].map(tracker._safe_normalize_team)
    stats["end"] = stats["game_id"].map(ends)
    assert stats["end"].notna().all(), "every 2025 PBP game must be timed"
    first_start_end = stats.groupby(["team", "passer_player_id"])["end"].min()
    final_passer = (stats.sort_values("end").groupby("team").tail(1).set_index("team"))[
        "passer_player_id"
    ]

    sample = games[
        (games["season"] == 2025) & (games["week"].isin(SAMPLED_2025_WEEKS))
    ].reset_index(drop=True)
    locks = lock_rule.lock_frame(sample)
    resolved = []
    for game in sample.to_dict("records"):
        lock = locks[game["game_id"]]
        game_end = (
            pd.Timestamp(game["kickoff_et"]).tz_convert("UTC") + DECLARED_GAME_DURATION
        )
        for column in ("home_team", "away_team"):
            team = tracker._safe_normalize_team(game[column])
            qbs = tracker.resolve_depth_chart_qbs(
                depth, 2025, int(game["week"]), team, lock
            )
            resolved.append(
                {
                    "game_id": game["game_id"],
                    "team": team,
                    "lock": lock,
                    "game_end": game_end,
                    "qb1": qbs.qb1,
                    "published_at": qbs.published_at,
                }
            )
    return {
        "resolved": pd.DataFrame(resolved),
        "first_start_end": first_start_end,
        "final_passer": final_passer,
        "depth": depth,
    }


class TestReal2025StarterIdentity:
    def test_the_2025_depth_charts_load_with_a_dt_column_and_no_week(
        self, real_2025
    ) -> None:
        depth = real_2025["depth"]
        assert "dt" in depth.columns
        assert "week" not in depth.columns, (
            "the loader must not manufacture a week for the dt schema"
        )
        assert isinstance(depth["dt"].dtype, pd.DatetimeTZDtype)

    def test_the_sample_is_non_empty_and_every_team_resolves(self, real_2025) -> None:
        resolved = real_2025["resolved"]
        assert len(resolved) >= 2 * 16 * len(SAMPLED_2025_WEEKS) - 40
        assert resolved["qb1"].notna().all()

    def test_every_snapshot_used_was_published_at_or_before_its_lock(
        self, real_2025
    ) -> None:
        resolved = real_2025["resolved"]
        for published, lock in zip(
            resolved["published_at"], resolved["lock"], strict=True
        ):
            assert published is not None
            assert lock_rule.is_admissible(published.to_pydatetime(), lock)

    def test_no_resolved_starter_first_started_in_a_later_game(self, real_2025) -> None:
        resolved = real_2025["resolved"]
        first = real_2025["first_start_end"]
        later = [
            (row.game_id, row.team, row.qb1)
            for row in resolved.itertuples()
            if (row.team, row.qb1) in first.index
            and first[(row.team, row.qb1)] > row.game_end
        ]
        assert later == [], f"hindsight starters: {later[:10]}"

    def test_non_vacuity_some_starter_is_not_the_seasons_final_passer(
        self, real_2025
    ) -> None:
        resolved = real_2025["resolved"]
        final = real_2025["final_passer"]
        differs = [
            row.game_id
            for row in resolved.itertuples()
            if row.qb1 != final.get(row.team)
        ]
        assert len(differs) > 0, (
            "if every resolved starter equals the season's final primary passer, the "
            "first-start check above could pass with the hindsight fallback still live"
        )


# ---------------------------------------------------------------------------
# Synthetic fixtures: two passers with different quality, one team
# ---------------------------------------------------------------------------

_STARTER_A = "00-000000A"
_BACKUP_B = "00-000000B"
_OPP = "00-000000C"


def _kickoff(day: int, hour: int = 13) -> pd.Timestamp:
    return pd.Timestamp(datetime(2025, 9, day, hour, 0, tzinfo=ET))


def _games() -> pd.DataFrame:
    """Weeks 1-2 history for KC and BUF, then a week-3 Sunday target KC vs BUF."""
    rows = [
        ("2025_W01_BUF@KC", 1, "KC", "BUF", _kickoff(7)),
        ("2025_W02_KC@BUF", 2, "BUF", "KC", _kickoff(14)),
        ("2025_W03_BUF@KC", 3, "KC", "BUF", _kickoff(21)),
    ]
    frame = pd.DataFrame(
        rows, columns=["game_id", "week", "home_team", "away_team", "kickoff_et"]
    )
    frame["season"] = 2025
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _plays(
    game: str,
    week: int,
    home: str,
    away: str,
    team: str,
    passer: str,
    epa: float,
    n: int = 20,
):
    return [
        {
            "game_id": game,
            "season": 2025,
            "week": week,
            "home_team": home,
            "away_team": away,
            "posteam": team,
            "passer_player_id": passer,
            "qb_epa": epa + i * 0.001,
            "cpoe": epa * 10 + i * 0.01,
        }
        for i in range(n)
    ]


def _pbp() -> pd.DataFrame:
    """A plays both prior weeks well, B poorly (in relief), the opponent in between."""
    plays = []
    plays += _plays("2025_01_BUF_KC", 1, "KC", "BUF", "KC", _STARTER_A, 0.4)
    plays += _plays("2025_01_BUF_KC", 1, "KC", "BUF", "BUF", _OPP, 0.1)
    plays += _plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _BACKUP_B, -0.3, n=10)
    plays += _plays("2025_02_KC_BUF", 2, "BUF", "KC", "BUF", _OPP, 0.1)
    plays += _plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _STARTER_A, 0.35)
    return pd.DataFrame(plays)


def _dt_depth(snapshots: list[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    """A dt-schema depth chart (loader-normalised names): one KC QB1/QB2 per snapshot."""
    rows = []
    for published, qb1 in snapshots:
        qb2 = _BACKUP_B if qb1 == _STARTER_A else _STARTER_A
        for rank, player in (("1", qb1), ("2", qb2)):
            rows.append(
                {
                    "dt": published,
                    "club_code": "KC",
                    "position": "QB",
                    "depth_team": rank,
                    "gsis_id": player,
                    "full_name": player,
                    "season": 2025,
                }
            )
        rows.append(
            {
                "dt": published,
                "club_code": "BUF",
                "position": "QB",
                "depth_team": "1",
                "gsis_id": _OPP,
                "full_name": _OPP,
                "season": 2025,
            }
        )
    frame = pd.DataFrame(rows)
    frame["dt"] = pd.to_datetime(frame["dt"], utc=True)
    return frame


def _tracker(depth: pd.DataFrame, pbp: pd.DataFrame | None = None) -> QBTracker:
    tracker = QBTracker()
    tracker._depth_chart_cache = {2025: depth}
    tracker._pbp_cache = {2024: pbp.iloc[0:0] if pbp is not None else _pbp().iloc[0:0]}
    tracker._pbp_cache[2025] = _pbp() if pbp is None else pbp
    return tracker


_TARGET = "2025_W03_BUF@KC"
_TARGET_LOCK = pd.Timestamp(lock_rule.game_lock(_kickoff(21)))


def _kc_adjustment(tracker: QBTracker) -> float:
    frame = tracker.build_features(
        _games(), _AS_OF_LATE, target_season=2025, target_week=3
    )
    row = frame.loc[(frame["game_id"] == _TARGET) & (frame["team"] == "KC")]
    assert len(row) == 1
    return float(row.iloc[0]["qb_adjustment"])


class TestTheDtBranchIsLoadBearing:
    """The binding instrument for the dt branch: a comment cannot fake this pair."""

    def _depth(self, planted_at: pd.Timestamp) -> pd.DataFrame:
        return _dt_depth(
            [(_TARGET_LOCK - timedelta(days=2), _STARTER_A), (planted_at, _BACKUP_B)]
        )

    def test_a_snapshot_one_second_after_the_lock_does_not_move_the_starter(
        self,
    ) -> None:
        tracker = _tracker(self._depth(_TARGET_LOCK + _ONE_SECOND))
        resolved = tracker.resolve_depth_chart_qbs(
            tracker._depth_chart_cache[2025], 2025, 3, "KC", _TARGET_LOCK
        )
        assert resolved.qb1 == _STARTER_A
        assert resolved.published_at == _TARGET_LOCK - timedelta(days=2)

    def test_the_same_snapshot_at_the_lock_does_move_the_starter(self) -> None:
        tracker = _tracker(self._depth(_TARGET_LOCK))
        resolved = tracker.resolve_depth_chart_qbs(
            tracker._depth_chart_cache[2025], 2025, 3, "KC", _TARGET_LOCK
        )
        assert resolved.qb1 == _BACKUP_B
        assert resolved.published_at == _TARGET_LOCK

    def test_the_built_feature_moves_only_with_the_at_lock_snapshot(self) -> None:
        baseline = _kc_adjustment(
            _tracker(_dt_depth([(_TARGET_LOCK - timedelta(days=2), _STARTER_A)]))
        )
        late = _kc_adjustment(_tracker(self._depth(_TARGET_LOCK + _ONE_SECOND)))
        at_lock = _kc_adjustment(_tracker(self._depth(_TARGET_LOCK)))

        assert late == baseline, "a post-lock snapshot must not move the value"
        assert at_lock != baseline, (
            "the at-lock snapshot must move it (positive control)"
        )


class TestTheWeekKeyedPath:
    def test_a_week_keyed_chart_resolves_that_week_never_a_later_one(self) -> None:
        depth = pd.DataFrame(
            {
                "season": [2024, 2024],
                "week": [3, 4],
                "club_code": ["KC", "KC"],
                "position": ["QB", "QB"],
                "depth_team": ["1", "1"],
                "gsis_id": [_STARTER_A, _BACKUP_B],
                "full_name": ["A", "B"],
            }
        )
        lock = pd.Timestamp(datetime(2024, 9, 21, 18, 0, tzinfo=ET))
        resolved = QBTracker().resolve_depth_chart_qbs(depth, 2024, 3, "KC", lock)
        assert resolved.qb1 == _STARTER_A
        assert resolved.published_at is None, (
            "a week-keyed chart has no publication time; none may be manufactured"
        )

    def test_a_real_2024_game_resolves_through_the_week_keyed_path(self) -> None:
        games = pd.read_parquet("data/silver/games.parquet")
        game = games[(games["season"] == 2024) & (games["week"] == 5)].iloc[0]
        lock = lock_rule.game_lock(game["kickoff_et"], game_id=str(game["game_id"]))
        tracker = QBTracker()
        depth = tracker._load_depth_charts(2024)
        assert "week" in depth.columns and "dt" not in depth.columns
        resolved = tracker.resolve_depth_chart_qbs(
            depth, 2024, 5, tracker._safe_normalize_team(game["home_team"]), lock
        )
        assert resolved.qb1 is not None
        assert resolved.published_at is None


# ---------------------------------------------------------------------------
# The play-by-play filter, per game
# ---------------------------------------------------------------------------


def _pbp_with_target_week() -> pd.DataFrame:
    """History plus a week-3 THURSDAY game (NYJ@NE) and the week-3 Sunday target itself."""
    frame = _pbp()
    extra = pd.DataFrame(
        _plays("2025_03_NYJ_NE", 3, "NE", "NYJ", "NE", "00-000000D", 0.2)
        + _plays("2025_03_BUF_KC", 3, "KC", "BUF", "KC", "00-000000Z", 0.9)
    )
    return pd.concat([frame, extra], ignore_index=True)


def _games_with_thursday() -> pd.DataFrame:
    games = _games()
    thursday = pd.DataFrame(
        [
            {
                "game_id": "2025_W03_NYJ@NE",
                "week": 3,
                "home_team": "NE",
                "away_team": "NYJ",
                "kickoff_et": pd.Timestamp(datetime(2025, 9, 18, 20, 15, tzinfo=ET)),
                "season": 2025,
            }
        ]
    )
    frame = pd.concat([games, thursday], ignore_index=True)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


class TestThePlayByPlayFilterIsPerGame:
    def test_a_finished_same_week_game_is_admitted_and_the_target_is_not(self) -> None:
        tracker = QBTracker()
        pbp = _pbp_with_target_week()
        ends = tracker.pbp_game_end_times(pbp, _games_with_thursday())
        admitted = set(tracker.admitted_pbp(pbp, ends, _TARGET_LOCK)["game_id"])

        assert "2025_03_NYJ_NE" in admitted, "the Thursday game ended before the lock"
        assert "2025_03_BUF_KC" not in admitted, "the target game's own plays"

    def test_the_fallback_passer_cannot_come_from_the_target_game(self) -> None:
        """No depth-chart row for KC: the fallback is the latest ADMITTED primary passer.

        The retired (season, week) fence admitted all of week 3 once the Thursday game had
        finished, so the fallback named Z, who appears only in the target game itself.
        """
        depth = _dt_depth([(_TARGET_LOCK - timedelta(days=2), _STARTER_A)])
        depth = depth[depth["club_code"] != "KC"]
        tracker = _tracker(depth, _pbp_with_target_week())
        frame = tracker.build_features(
            _games_with_thursday(), _AS_OF_LATE, target_season=2025, target_week=3
        )
        kc = frame.loc[(frame["game_id"] == _TARGET) & (frame["team"] == "KC")]
        with_a = _kc_adjustment(
            _tracker(_dt_depth([(_TARGET_LOCK - timedelta(days=2), _STARTER_A)]))
        )
        assert float(kc.iloc[0]["qb_adjustment"]) == with_a, (
            "the fallback must name A (last admitted KC primary passer, week 2), not Z"
        )

    def test_no_frame_wide_scalar_cutoff_remains(self) -> None:
        source = inspect.getsource(qb_module)
        assert "pd.Timestamp(as_of_datetime)" not in source


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestTheQbProvenance:
    def test_the_tracker_is_an_information_time_provider(self) -> None:
        assert isinstance(QBTracker(), InformationTimeProvider)
        assert dict(QBTracker().no_information_signature()) == {"qb_adjustment": 0.0}

    def test_the_time_is_the_max_of_the_snapshot_and_the_latest_admitted_game(
        self,
    ) -> None:
        snapshot = _TARGET_LOCK - timedelta(hours=3)
        tracker = _tracker(_dt_depth([(snapshot, _STARTER_A)]))
        provenance = tracker.information_times(
            _games(), target_season=2025, target_week=3
        )
        assert list(provenance.columns) == list(PROVENANCE_COLUMNS)
        row = provenance.loc[provenance["game_id"] == _TARGET].iloc[0]
        assert row["basis"] == InformationBasis.PER_ROW.value
        assert pd.Timestamp(row["information_time"]) == snapshot

    def test_the_gate_accepts_the_synthetic_build(self) -> None:
        tracker = _tracker(_dt_depth([(_TARGET_LOCK - timedelta(days=2), _STARTER_A)]))
        games = _games()
        frame = tracker.build_features(games, _AS_OF_LATE)
        provenance = tracker.information_times(games)
        state = InformationTimeGate().check(
            "qb_tracking",
            frame,
            provenance,
            lock_rule.lock_frame(games),
            no_information_signature=tracker.no_information_signature(),
        )
        assert state is SourceCheckState.CHECKED
        week_one = provenance.loc[provenance["game_id"] == "2025_W01_BUF@KC"].iloc[0]
        assert week_one["basis"] == InformationBasis.NO_INFORMATION.value, (
            "week 1 has no snapshot at its lock and no finished game: an honest unknown"
        )


# ---------------------------------------------------------------------------
# The week-zero scan, structural, with both controls
# ---------------------------------------------------------------------------


def _is_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and node.value == 0
        and not isinstance(node.value, bool)
    )


def _is_week(node: ast.AST) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id == "week")
        or (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "week"
        )
        or (isinstance(node, ast.Attribute) and node.attr == "week")
    )


def week_zero_nodes(source: str) -> list[int]:
    """Line numbers of any node binding or comparing ``week`` to the literal ``0``.

    Structural, on the parsed tree: comments never reach the AST and a docstring is a
    string constant, so an explanation of the deleted assignment is never a hit.
    """
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_zero(node.value):
            lines += [node.lineno for target in node.targets if _is_week(target)]
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_zero(node.value)
            and _is_week(node.target)
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            lines += [
                node.lineno
                for keyword in node.keywords
                if keyword.arg == "week" and _is_zero(keyword.value)
            ]
        elif (
            isinstance(node, ast.Compare)
            and _is_week(node.left)
            and any(_is_zero(c) for c in node.comparators)
        ):
            lines.append(node.lineno)
    return sorted(lines)


class TestTheWeekZeroAssignmentIsGone:
    def test_control_a_planted_assignment_is_flagged(self) -> None:
        planted = 'def f(dc):\n    dc["week"] = 0\n    week = 0\n    return week\n'
        assert week_zero_nodes(planted) == [2, 3]

    def test_control_an_explanation_in_a_comment_and_docstring_is_not_flagged(
        self,
    ) -> None:
        explained = (
            "def load(dc):\n"
            '    """The old loader set week = 0 for the dt schema; that is deleted."""\n'
            '    # dc["week"] = 0 used to live here; the dt branch replaced it.\n'
            "    return dc\n"
        )
        assert week_zero_nodes(explained) == []

    def test_the_qb_module_binds_no_week_to_zero(self) -> None:
        assert week_zero_nodes(inspect.getsource(qb_module)) == []
