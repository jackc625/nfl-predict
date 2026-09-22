"""THE shared reveal-test harness for every builder fenced at its game's own lock.

Plan 33.2-13 Task 3 (SPEC R5 / R2). A lock fence that admits everything looks exactly like
a lock fence that works until something is PLANTED against it, and a provenance derived
with the same rule the builder used proves the rule, not the content (RESEARCH P3). So each
builder registered here is driven through the same three assertions:

1. a row planted one second AFTER the target game's lock leaves the produced feature
   values byte-identical to the build without it;
2. the same row planted exactly AT the lock changes at least one value (the positive
   control -- it is what makes assertion 1 mean "fenced" rather than "inert fixture");
3. when the planted row's information time is reported honestly, the information-time
   gate REFUSES the frame and names the game.

ONE HARNESS, NOT SIX. ``REVEAL_CASES`` is seeded here with ``injury`` and ``qb``. Plan
33.2-14 adds ``contextual``, ``market_anchors``, ``weather`` and ``snaps`` to the same
tuple and raises ``DECLARED_REVEAL_CASE_COUNT``; a builder silently leaving the list fails
the declared-length control.

Structural controls (the ``tests/unit/test_freeze_parse_single_source.py`` shape): the
parameter list has an asserted length; a synthetic builder that IGNORES the cutoff is
flagged; a synthetic builder that honours it is not.

The same module proves the gate's refusals survive the PRODUCTION ORCHESTRATION.
``scripts/build_features.py`` turns every member of ``_SOURCE_LOAD_ERRORS`` into an EMPTY
source frame, so the refusals are excluded BY TYPE (asserted member by member, on the
``tests/unit/test_provisional_training_refusal.py`` pattern) and then driven through
``load_all_feature_sources`` and ``generate_feature_matrices`` to show they reach the
caller. Nothing here writes ``data/`` or ``artifacts/``: every write lands under
``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.injury import InjuryBuilder
from features.provenance import (
    InformationBasis,
    InformationTimeGate,
    InformationTimeViolation,
    ProvenanceCoverageError,
    SourceCheckState,
    UndatedSourceError,
)
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder

ET = ZoneInfo("America/New_York")
_ONE_SECOND = timedelta(seconds=1)
_AS_OF = datetime(2030, 1, 1, tzinfo=ET)


# ---------------------------------------------------------------------------
# The scenario contract every reveal case supplies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevealScenario:
    """One builder, one target game, and a way to plant a row at a chosen instant.

    Attributes:
        source_name: The ``feature_sources`` registry key the gate names.
        games: The games frame the build runs over.
        game_id: The target game.
        feature_columns: The produced columns the reveal compares.
        build: ``planted_at -> (source frame, provenance, signature)``; ``None`` plants
            nothing.
    """

    source_name: str
    games: pd.DataFrame
    game_id: str
    feature_columns: tuple[str, ...]
    build: Callable[
        [pd.Timestamp | None],
        tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]],
    ]

    @property
    def lock(self) -> pd.Timestamp:
        return pd.Timestamp(lock_rule.lock_frame(self.games)[self.game_id])

    def values(self, planted_at: pd.Timestamp | None) -> pd.DataFrame:
        frame, _, _ = self.build(planted_at)
        return (
            frame.loc[
                frame["game_id"] == self.game_id, ["game_id", *self.feature_columns]
            ]
            .sort_values(["game_id", *self.feature_columns])
            .reset_index(drop=True)
        )


class RevealCase(NamedTuple):
    """``(name, scenario factory, feature column)``: the harness's parameter shape."""

    name: str
    scenario: Callable[[], RevealScenario]
    feature_column: str


def reveal_violations(scenario: RevealScenario) -> list[str]:
    """What is wrong with *scenario*'s fence, or ``[]`` when it is load-bearing.

    The harness's one judgement, shared by every registered builder and by the two
    structural controls, so the controls exercise exactly the code the cases rely on.
    """
    problems: list[str] = []
    baseline = scenario.values(None)
    late = scenario.values(scenario.lock + _ONE_SECOND)
    at_lock = scenario.values(scenario.lock)
    if not late.equals(baseline):
        problems.append("a row planted at lock+1s changed the produced values")
    if at_lock.equals(baseline):
        problems.append(
            "the same row planted AT the lock changed nothing: the fence is not "
            "load-bearing (or the planted row reaches no feature)"
        )
    return problems


def honest_late_provenance(scenario: RevealScenario) -> pd.DataFrame:
    """The provenance a builder that USED the lock+1s row would have to report."""
    _, provenance, _ = scenario.build(scenario.lock)
    provenance = provenance.copy()
    target = provenance["game_id"] == scenario.game_id
    provenance["information_time"] = provenance["information_time"].astype(object)
    provenance.loc[target, "basis"] = InformationBasis.PER_ROW.value
    provenance.loc[target, "information_time"] = scenario.lock + _ONE_SECOND
    return provenance


# ---------------------------------------------------------------------------
# Case 1: injury -- a QB1 "Out" report planted against a Thursday game
# ---------------------------------------------------------------------------

_KC_QB1 = "00-0033873"
_KC_QB2 = "00-0000002"


def _empty_snap_builder() -> SnapCountBuilder:
    """A hermetic SnapCountBuilder: one week-1 row, so week-1 targets have no prior share."""
    return SnapCountBuilder(
        snaps_df=pd.DataFrame(
            {
                "game_id": ["2023_01_BUF_KC"],
                "pfr_player_id": ["kc_qb"],
                "player": ["kc_qb"],
                "position": ["QB"],
                "team": ["KC"],
                "opponent": ["BUF"],
                "season": [2023],
                "week": [1],
                "offense_snaps": [70.0],
                "offense_pct": [1.0],
                "defense_snaps": [0.0],
                "defense_pct": [0.0],
                "st_snaps": [0.0],
                "st_pct": [0.0],
            }
        )
    )


def _injury_scenario() -> RevealScenario:
    games = pd.DataFrame(
        [
            {
                "game_id": "2023_W01_BUF@KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": pd.Timestamp(datetime(2023, 9, 7, 20, 20, tzinfo=ET)),
            }
        ]
    )
    games["kickoff_et"] = pd.to_datetime(games["kickoff_et"], utc=True)
    lock = pd.Timestamp(lock_rule.lock_frame(games).iloc[0])
    depth = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "club_code": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_team": ["1", "2"],
            "gsis_id": [_KC_QB1, _KC_QB2],
            "full_name": ["QB1", "QB2"],
        }
    )

    def build(planted_at: pd.Timestamp | None):
        rows = [
            {
                "season": 2023,
                "week": 1,
                "team": "KC",
                "gsis_id": _KC_QB1,
                "position": "QB",
                "report_status": "Questionable",
                "date_modified": lock - timedelta(days=2),
            }
        ]
        if planted_at is not None:
            rows.append(
                {**rows[0], "report_status": "Out", "date_modified": planted_at}
            )
        injuries = pd.DataFrame(rows)
        injuries["date_modified"] = pd.to_datetime(injuries["date_modified"], utc=True)
        builder = InjuryBuilder(
            snap_builder=_empty_snap_builder(),
            injuries_df=injuries,
            depth_charts_df=depth,
            pbp_df=pd.DataFrame(),
        )
        frame = builder.build_features(games, _AS_OF)
        return (
            frame,
            builder.information_times(games),
            dict(builder.no_information_signature()),
        )

    return RevealScenario(
        source_name="injury",
        games=games,
        game_id="2023_W01_BUF@KC",
        feature_columns=("home_qb_out_flag",),
        build=build,
    )


# ---------------------------------------------------------------------------
# Case 2: qb -- a 2025 depth-chart snapshot naming a different QB1
# ---------------------------------------------------------------------------

_A, _B, _OPP = "00-000000A", "00-000000B", "00-000000C"


def _qb_plays(game, week, home, away, team, passer, epa, n=20):
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


def _qb_scenario() -> RevealScenario:
    kickoffs = [datetime(2025, 9, day, 13, 0, tzinfo=ET) for day in (7, 14, 21)]
    games = pd.DataFrame(
        {
            "game_id": ["2025_W01_BUF@KC", "2025_W02_KC@BUF", "2025_W03_BUF@KC"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "home_team": ["KC", "BUF", "KC"],
            "away_team": ["BUF", "KC", "BUF"],
            "kickoff_et": pd.to_datetime(kickoffs, utc=True),
        }
    )
    lock = pd.Timestamp(lock_rule.lock_frame(games)["2025_W03_BUF@KC"])
    pbp = pd.DataFrame(
        _qb_plays("2025_01_BUF_KC", 1, "KC", "BUF", "KC", _A, 0.4)
        + _qb_plays("2025_01_BUF_KC", 1, "KC", "BUF", "BUF", _OPP, 0.1)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _A, 0.35)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _B, -0.3, n=10)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "BUF", _OPP, 0.1)
    )

    def snapshot(published: pd.Timestamp, kc_qb1: str) -> list[dict]:
        other = _B if kc_qb1 == _A else _A
        return [
            {"dt": published, "club_code": "KC", "position": "QB", "depth_team": "1",
             "gsis_id": kc_qb1, "full_name": kc_qb1, "season": 2025},
            {"dt": published, "club_code": "KC", "position": "QB", "depth_team": "2",
             "gsis_id": other, "full_name": other, "season": 2025},
            {"dt": published, "club_code": "BUF", "position": "QB", "depth_team": "1",
             "gsis_id": _OPP, "full_name": _OPP, "season": 2025},
        ]  # fmt: skip

    def build(planted_at: pd.Timestamp | None):
        rows = snapshot(lock - timedelta(days=2), _A)
        if planted_at is not None:
            rows += snapshot(planted_at, _B)
        depth = pd.DataFrame(rows)
        depth["dt"] = pd.to_datetime(depth["dt"], utc=True)
        tracker = QBTracker()
        tracker._depth_chart_cache = {2025: depth}
        tracker._pbp_cache = {2024: pbp.iloc[0:0], 2025: pbp}
        frame = tracker.build_features(games, _AS_OF)
        return (
            frame,
            tracker.information_times(games),
            dict(tracker.no_information_signature()),
        )

    return RevealScenario(
        source_name="qb_tracking",
        games=games,
        game_id="2025_W03_BUF@KC",
        feature_columns=("team", "qb_adjustment"),
        build=build,
    )


# ---------------------------------------------------------------------------
# THE PARAMETER LIST. Plan 33.2-14 appends contextual, market_anchors, weather and snaps
# here and raises the declared count to 6.
# ---------------------------------------------------------------------------

REVEAL_CASES: tuple[RevealCase, ...] = (
    RevealCase("injury", _injury_scenario, "home_qb_out_flag"),
    RevealCase("qb", _qb_scenario, "qb_adjustment"),
)
DECLARED_REVEAL_CASE_COUNT = 2


class TestTheHarnessIsNotVacuous:
    def test_the_parameter_list_has_its_declared_length(self) -> None:
        assert len(REVEAL_CASES) == DECLARED_REVEAL_CASE_COUNT
        assert len({case.name for case in REVEAL_CASES}) == DECLARED_REVEAL_CASE_COUNT

    def test_every_case_names_a_column_its_scenario_compares(self) -> None:
        for case in REVEAL_CASES:
            assert case.feature_column in case.scenario().feature_columns


@pytest.mark.parametrize("case", REVEAL_CASES, ids=lambda case: case.name)
class TestEveryRegisteredBuilderIsFencedAtItsLock:
    def test_a_row_planted_one_second_after_the_lock_changes_nothing(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        pd.testing.assert_frame_equal(
            scenario.values(scenario.lock + _ONE_SECOND),
            scenario.values(None),
            check_exact=True,
        )

    def test_the_same_row_planted_at_the_lock_changes_a_value(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        assert not scenario.values(scenario.lock).equals(scenario.values(None))

    def test_the_harness_judgement_passes(self, case: RevealCase) -> None:
        assert reveal_violations(case.scenario()) == []

    def test_the_gate_accepts_what_the_builder_honestly_reports(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        frame, provenance, signature = scenario.build(scenario.lock + _ONE_SECOND)
        state = InformationTimeGate().check(
            scenario.source_name,
            frame,
            provenance,
            lock_rule.lock_frame(scenario.games),
            no_information_signature=signature,
        )
        assert state is SourceCheckState.CHECKED

    def test_the_gate_refuses_the_planted_row_reported_honestly(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        frame, _, signature = scenario.build(scenario.lock)
        with pytest.raises(InformationTimeViolation) as refused:
            InformationTimeGate().check(
                scenario.source_name,
                frame,
                honest_late_provenance(scenario),
                lock_rule.lock_frame(scenario.games),
                no_information_signature=signature,
            )
        assert refused.value.details["game_ids"] == [scenario.game_id]
        assert refused.value.details["violation_type"] == "information_time"


# ---------------------------------------------------------------------------
# Structural controls: the harness flags a cutoff-ignoring builder, not a compliant one
# ---------------------------------------------------------------------------


def _counting_scenario(*, honours_the_lock: bool) -> RevealScenario:
    """A synthetic builder whose one feature counts the rows it admitted."""
    games = pd.DataFrame(
        {
            "game_id": ["2024_W05_NE@NYJ"],
            "season": [2024],
            "week": [5],
            "home_team": ["NYJ"],
            "away_team": ["NE"],
            "kickoff_et": pd.to_datetime(
                [datetime(2024, 10, 6, 13, 0, tzinfo=ET)], utc=True
            ),
        }
    )
    lock = pd.Timestamp(lock_rule.lock_frame(games).iloc[0])

    def build(planted_at: pd.Timestamp | None):
        times = [lock - timedelta(days=1)]
        if planted_at is not None:
            times.append(planted_at)
        admitted = (
            [t for t in times if lock_rule.is_admissible(t, lock)]
            if honours_the_lock
            else times
        )
        frame = pd.DataFrame(
            {"game_id": ["2024_W05_NE@NYJ"], "rows_used": [float(len(admitted))]}
        )
        provenance = pd.DataFrame(
            {
                "game_id": ["2024_W05_NE@NYJ"],
                "basis": [InformationBasis.PER_ROW.value],
                "information_time": [max(admitted)],
            }
        )
        return frame, provenance, {"rows_used": 0.0}

    return RevealScenario(
        source_name="synthetic",
        games=games,
        game_id="2024_W05_NE@NYJ",
        feature_columns=("rows_used",),
        build=build,
    )


class TestTheHarnessControls:
    def test_control_a_builder_that_ignores_the_cutoff_is_flagged(self) -> None:
        problems = reveal_violations(_counting_scenario(honours_the_lock=False))
        assert problems == ["a row planted at lock+1s changed the produced values"]

    def test_control_a_builder_that_honours_the_cutoff_is_not_flagged(self) -> None:
        assert reveal_violations(_counting_scenario(honours_the_lock=True)) == []


# ---------------------------------------------------------------------------
# The refusals survive the production orchestration
# ---------------------------------------------------------------------------

_REFUSALS: dict[str, type[Exception]] = {
    "InformationTimeViolation": InformationTimeViolation,
    "UndatedSourceError": UndatedSourceError,
    "ProvenanceCoverageError": ProvenanceCoverageError,
}


class TestTheRefusalsAreExcludedFromTheSwallowTupleByType:
    def test_the_swallow_tuple_is_not_empty(self) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        assert len(_SOURCE_LOAD_ERRORS) > 0, (
            "an empty tuple would make every issubclass check below pass vacuously"
        )

    @pytest.mark.parametrize("refusal_name", sorted(_REFUSALS))
    def test_no_refusal_is_a_member_or_a_subclass_of_a_member(
        self, refusal_name: str
    ) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        refusal = _REFUSALS[refusal_name]
        assert refusal not in _SOURCE_LOAD_ERRORS
        for caught in _SOURCE_LOAD_ERRORS:
            assert not issubclass(refusal, caught), (
                f"{refusal_name} is a subclass of {caught.__name__}, which is in "
                "_SOURCE_LOAD_ERRORS -- the injury guard would turn the refusal into an "
                "empty injury frame and the build would exit green"
            )


def _sandbox_builder(monkeypatch, tmp_path):
    """A FeatureMatrixBuilder over a sandboxed silver ``games`` table (one 2026 week)."""
    from data.storage import save_dataframe
    from scripts.build_features import FeatureMatrixBuilder
    from tests.fixtures.elo_sandbox import (
        make_season_games,
        redirect_storage_to_sandbox,
    )

    sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
    games = make_season_games(2026, weeks=1)
    save_dataframe(games, "games", layer="silver", replace_mode=True)
    builder = FeatureMatrixBuilder()
    # The QB source is not under test here and reads pinned upstream play-by-play;
    # an empty frame is reported EMPTY_UNCHECKED by the gate and checks nothing.
    monkeypatch.setattr(
        builder.qb_tracker,
        "build_features",
        lambda *_a, **_k: pd.DataFrame(columns=["game_id", "team", "qb_adjustment"]),
    )
    # The injury builder draws its availability weights from the snap builder, and the
    # sandbox has no snap table: without this frame the injury source would fail to load
    # and become EMPTY, and the build would run on past Stage 1.
    builder.snap_builder._snaps_df = _empty_snap_builder()._snaps_df
    # LeakageGate.write_diagnostic_report defaults to the PRODUCTION outputs/diagnostics
    # tree. Any refusal this module provokes past Stage 1 must write under tmp_path.
    real_writer = builder.leakage_gate.write_diagnostic_report
    monkeypatch.setattr(
        builder.leakage_gate,
        "write_diagnostic_report",
        lambda violation, output_dir=None: real_writer(
            violation, output_dir=str(tmp_path / "diagnostics")
        ),
    )
    return sandbox, games, builder


def _injured_week(games: pd.DataFrame) -> pd.DataFrame:
    """One dated, pre-lock report per game so the injury frame carries real rows."""
    locks = lock_rule.lock_frame(games)
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "team": game["home_team"],
                "gsis_id": f"00-00000{index:02d}",
                "position": "WR",
                "report_status": "Out",
                "date_modified": locks[game["game_id"]] - timedelta(days=1),
            }
            for index, game in enumerate(games.to_dict("records"))
        ]
    )
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


class TestTheRefusalReachesTheCaller:
    def test_control_an_ordinary_injury_load_error_becomes_an_empty_frame(
        self, tmp_path, monkeypatch
    ) -> None:
        """The CONTROL: the swallow path exists and fires for a tuple member."""
        _, _, builder = _sandbox_builder(monkeypatch, tmp_path)

        def _raise(*_a, **_k):
            raise ValueError("an ordinary source-load failure")

        monkeypatch.setattr(builder.injury_builder, "build_features", _raise)
        sources = builder.load_all_feature_sources(target_season=2026)
        assert len(sources["injury"]) == 0

    def test_a_refusal_from_the_injury_builder_escapes_load_all_feature_sources(
        self, tmp_path, monkeypatch
    ) -> None:
        _, _, builder = _sandbox_builder(monkeypatch, tmp_path)

        def _refuse(*_a, **_k):
            raise InformationTimeViolation(
                "planted: an injury value built after its lock",
                {"source": "injury", "game_ids": ["2026_W01_MIA@BUF"]},
            )

        monkeypatch.setattr(builder.injury_builder, "build_features", _refuse)
        with pytest.raises(InformationTimeViolation):
            builder.load_all_feature_sources(target_season=2026)

    def test_a_post_lock_injury_time_stops_the_build_and_writes_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The production path: the real injury builder's frame, a provenance that reports
        one game one second after its lock, and the real Stage-1 gate inside
        ``generate_feature_matrices``. The refusal must reach the caller -- not an empty
        injury frame and a green exit."""
        sandbox, games, builder = _sandbox_builder(monkeypatch, tmp_path)
        builder.injury_builder._injuries_df = _injured_week(games)
        target = str(games["game_id"].iloc[0])
        late = pd.Timestamp(lock_rule.lock_frame(games)[target]) + _ONE_SECOND
        honest = builder.injury_builder.information_times

        def _late_provenance(games_df, **kwargs):
            provenance = honest(games_df, **kwargs).copy()
            provenance["information_time"] = provenance["information_time"].astype(
                object
            )
            hit = provenance["game_id"] == target
            provenance.loc[hit, "basis"] = InformationBasis.PER_ROW.value
            provenance.loc[hit, "information_time"] = late
            return provenance

        monkeypatch.setattr(
            builder.injury_builder, "information_times", _late_provenance
        )
        with pytest.raises(InformationTimeViolation) as refused:
            builder.generate_feature_matrices(target_season=2026)

        assert refused.value.details["source"] == "injury"
        assert refused.value.details["game_ids"] == [target]
        assert not list((sandbox / "gold").glob("*.parquet")), "nothing may be written"
