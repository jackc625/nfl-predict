"""Leakage proof for the injury signal, retargeted at EACH GAME'S OWN LOCK (Plan 33.2-13).

REWRITTEN by Plan 33.2-13 Task 1. The module used to prove the Phase-28 Friday-freeze
fence (``date_modified <= as_of_datetime``, one cutoff for the whole frame). That cutoff
is ``datetime.now(ET)`` in production, which admits everything that already happened, so
its three proofs were retargeted rather than deleted. Every intent survives:

  1. TIME FENCE -- a report is admitted only when its information time is AT or BEFORE the
     target game's lock (18:00 ET on the ET calendar day before kickoff, ``utils.game_lock``).
     At-lock is admitted; one second later is not.
  2. WITHHOLD-FUTURE BYTE IDENTITY -- revealing a post-lock report leaves every feature
     byte-identical to the build without it.
  3. POSITIVE CONTROL -- the same report moved to exactly the lock DOES change the value,
     so the fence (not an inert fixture) is what protects the withhold test.

ADDED by the same task:

  * the latest report ADMITTED at or before the lock wins, not the latest overall;
  * an undatable row (2009-era null ``date_modified``) is excluded and its game flagged;
  * the 2025-shaped frame (no ``date_modified`` column at all) resolves to an honest unknown,
    and the capture-stamp column admits a row only when its stamp is at or before the lock;
  * the provenance the builder reports is never dishonest: no ``per_row`` row with a null
    time, so ``UndatedSourceError`` is unreachable from this builder (the 13 -> 14 -> 15
    inversion fix; Plan 33.2-14's full-history rebuild depends on it);
  * the docstring-excluding scanner that pins the removal of the week-keyed fallback, with a
    planted-violation control and a no-false-positive control.

Nothing here reads or writes ``data/``: every frame is injected.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import features.injury as injury_module
import utils.game_lock as lock_rule
from features.injury import (
    UPSTREAM_CAPTURE_COLUMN,
    InjuryBuilder,
    InjurySelection,
    SelectionRule,
)
from features.protocol import InformationTimeProvider
from features.provenance import (
    PROVENANCE_COLUMNS,
    InformationBasis,
    InformationTimeGate,
    SourceCheckState,
)
from features.snaps import SnapCountBuilder

ET = ZoneInfo("America/New_York")

# Mahomes (the KC QB1) and a backup gsis_id used across the fixtures.
_QB1_ID = "00-0033873"
_QB2_ID = "00-0000002"

# A THURSDAY kickoff, so the lock is the Wednesday: the retired Friday freeze would have
# admitted Thursday-morning reports for this game, which is the shape the per-game lock ends.
_GAME_ID = "2023_W01_BUF@KC"
_KICKOFF = datetime(2023, 9, 7, 20, 20, tzinfo=ET)
_LOCK = pd.Timestamp(lock_rule.game_lock(_KICKOFF))

# A SUNDAY game in the same week, for the per-game (not per-frame) proof.
_SUNDAY_GAME_ID = "2023_W01_MIA@LAC"
_SUNDAY_KICKOFF = datetime(2023, 9, 10, 16, 25, tzinfo=ET)
_SUNDAY_LOCK = pd.Timestamp(lock_rule.game_lock(_SUNDAY_KICKOFF))

_ONE_SECOND = timedelta(seconds=1)

# The as_of argument the FeatureBuilder Protocol still carries. It is deliberately AFTER
# every fixture time: it is not the fence, so it must not be what keeps a row out.
_AS_OF_LATE = datetime(2023, 12, 31, 12, 0, tzinfo=ET)

_FEATURE_COLUMNS = [
    "home_qb_out_flag",
    "home_backup_quality_delta",
    "home_availability_fraction",
    "home_injury_coverage",
    "home_date_modified_coverage",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _injury_row(
    gsis_id: str,
    position: str,
    status: str | None,
    date_modified: pd.Timestamp | None,
    team: str = "KC",
    season: int = 2023,
) -> dict:
    """One injury silver row (shape of ``nfl.load_injuries(season).to_pandas()``)."""
    return {
        "season": season,
        "week": 1,
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "full_name": "Player",
        "report_status": status,
        "date_modified": date_modified,
    }


def _injuries(rows: list[dict]) -> pd.DataFrame:
    """An injuries frame whose ``date_modified`` is UTC-aware, exactly as silver stores it."""
    frame = pd.DataFrame(rows)
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


def _depth_charts(season: int = 2023) -> pd.DataFrame:
    """Week-keyed depth charts: Mahomes KC QB1, a backup KC QB2."""
    return pd.DataFrame(
        {
            "season": [season, season],
            "week": [1, 1],
            "club_code": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_team": ["1", "2"],
            "gsis_id": [_QB1_ID, _QB2_ID],
            "full_name": ["Patrick Mahomes", "Backup QB"],
        }
    )


def _snap_builder() -> SnapCountBuilder:
    """A hermetic SnapCountBuilder: week-1 targets have no prior shares."""
    snaps = pd.DataFrame(
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
    return SnapCountBuilder(snaps_df=snaps)


def _games(*, with_sunday: bool = False, season: int = 2023) -> pd.DataFrame:
    """The Thursday KC (home) vs BUF (away) game, optionally with a Sunday LAC game."""
    rows = [
        {
            "game_id": _GAME_ID.replace("2023", str(season), 1),
            "season": season,
            "week": 1,
            "home_team": "KC",
            "away_team": "BUF",
            "kickoff_et": _KICKOFF.replace(year=season),
        }
    ]
    if with_sunday:
        rows.append(
            {
                "game_id": _SUNDAY_GAME_ID,
                "season": season,
                "week": 1,
                "home_team": "LAC",
                "away_team": "MIA",
                "kickoff_et": _SUNDAY_KICKOFF,
            }
        )
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def _builder(injuries: pd.DataFrame, *, season: int = 2023) -> InjuryBuilder:
    """An InjuryBuilder on the fixtures, with an empty PBP frame (no nflreadpy access)."""
    return InjuryBuilder(
        snap_builder=_snap_builder(),
        injuries_df=injuries,
        depth_charts_df=_depth_charts(season),
        pbp_df=pd.DataFrame(),
    )


def _build(injuries: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.DataFrame:
    games = _games() if games is None else games
    return _builder(injuries).build_features(
        games, _AS_OF_LATE, target_season=2023, target_week=1
    )


def _game_row(games: pd.DataFrame, game_id: str) -> dict:
    return games.loc[games["game_id"] == game_id].iloc[0].to_dict()


def _lock_frame(games: pd.DataFrame) -> pd.Series:
    return lock_rule.lock_frame(games)


def _gate_check(builder: InjuryBuilder, games: pd.DataFrame) -> SourceCheckState:
    frame = builder.build_features(
        games, _AS_OF_LATE, target_season=2023, target_week=1
    )
    provenance = builder.information_times(games, target_season=2023, target_week=1)
    return InformationTimeGate().check(
        "injury",
        frame,
        provenance,
        _lock_frame(games),
        no_information_signature=builder.no_information_signature(),
    )


# ---------------------------------------------------------------------------
# 1. The time fence, on each game's own lock
# ---------------------------------------------------------------------------


class TestThePerGameLockFence:
    """Intent 1: admitted iff the information time is at or before THIS game's lock."""

    def test_a_report_exactly_at_the_lock_is_admitted(self) -> None:
        games = _games()
        builder = _builder(_injuries([_injury_row(_QB1_ID, "QB", "Out", _LOCK)]))

        selection = builder.fenced_injuries(_game_row(games, _GAME_ID), _LOCK)

        assert isinstance(selection, InjurySelection)
        assert selection.rule is SelectionRule.DATE_MODIFIED
        assert len(selection.rows) == 1
        assert selection.information_time == _LOCK

    def test_a_report_one_second_after_the_lock_is_not_admitted(self) -> None:
        games = _games()
        late = _LOCK + _ONE_SECOND
        builder = _builder(_injuries([_injury_row(_QB1_ID, "QB", "Out", late)]))

        selection = builder.fenced_injuries(_game_row(games, _GAME_ID), _LOCK)

        assert selection.rule is SelectionRule.NONE_ADMITTED
        assert len(selection.rows) == 0
        assert selection.information_time is None

    def test_each_game_is_fenced_at_its_own_lock_not_one_frame_wide_cutoff(
        self,
    ) -> None:
        """A Thursday-noon report sits AFTER the Thursday game's Wednesday lock and BEFORE
        the Sunday game's Saturday lock: the same frame admits it for one game only."""
        games = _games(with_sunday=True)
        thursday_noon = pd.Timestamp(datetime(2023, 9, 7, 12, 0, tzinfo=ET))
        assert _LOCK < thursday_noon < _SUNDAY_LOCK
        injuries = _injuries(
            [
                _injury_row(_QB1_ID, "QB", "Out", thursday_noon, team="KC"),
                _injury_row("00-0000009", "WR", "Out", thursday_noon, team="LAC"),
            ]
        )
        builder = _builder(injuries)

        thursday = builder.fenced_injuries(_game_row(games, _GAME_ID), _LOCK)
        sunday = builder.fenced_injuries(
            _game_row(games, _SUNDAY_GAME_ID), _SUNDAY_LOCK
        )

        assert thursday.rule is SelectionRule.NONE_ADMITTED
        assert sunday.rule is SelectionRule.DATE_MODIFIED
        assert set(sunday.rows["team"]) == {"LAC"}


class TestWithholdFutureByteIdentity:
    """Intents 2 and 3: the reveal test and its positive control, on the per-game lock."""

    @staticmethod
    def _base_rows() -> list[dict]:
        return [_injury_row(_QB1_ID, "QB", "Questionable", _LOCK - timedelta(days=2))]

    def test_revealing_a_post_lock_report_leaves_every_feature_byte_identical(
        self,
    ) -> None:
        withheld = _build(_injuries(self._base_rows()))
        revealed = _build(
            _injuries(
                [
                    *self._base_rows(),
                    _injury_row(_QB1_ID, "QB", "Out", _LOCK + _ONE_SECOND),
                ]
            )
        )

        pd.testing.assert_frame_equal(
            withheld[_FEATURE_COLUMNS], revealed[_FEATURE_COLUMNS], check_exact=True
        )
        assert float(revealed.iloc[0]["home_qb_out_flag"]) == 0.0

    def test_positive_control_the_same_report_at_the_lock_changes_the_value(
        self,
    ) -> None:
        withheld = _build(_injuries(self._base_rows()))
        at_lock = _build(
            _injuries([*self._base_rows(), _injury_row(_QB1_ID, "QB", "Out", _LOCK)])
        )

        assert float(withheld.iloc[0]["home_qb_out_flag"]) == 0.0
        assert float(at_lock.iloc[0]["home_qb_out_flag"]) == 1.0

    def test_the_protocol_as_of_argument_is_not_the_fence(self) -> None:
        """An as_of FAR after the post-lock report still does not admit it: the lock does
        the fencing, so a frame-wide cutoff cannot be what the proof rests on."""
        revealed = _builder(
            _injuries(
                [
                    *self._base_rows(),
                    _injury_row(_QB1_ID, "QB", "Out", _LOCK + _ONE_SECOND),
                ]
            )
        ).build_features(
            _games(),
            datetime(2030, 1, 1, tzinfo=ET),
            target_season=2023,
            target_week=1,
        )
        assert float(revealed.iloc[0]["home_qb_out_flag"]) == 0.0


# ---------------------------------------------------------------------------
# 2. The latest ADMITTED report wins
# ---------------------------------------------------------------------------


class TestTheLatestAdmittedReportWins:
    def test_three_reports_resolve_to_the_latest_one_at_or_before_the_lock(
        self,
    ) -> None:
        games = _games()
        injuries = _injuries(
            [
                _injury_row(_QB1_ID, "QB", "Out", _LOCK - timedelta(days=2)),
                _injury_row(_QB1_ID, "QB", "Questionable", _LOCK - timedelta(hours=1)),
                _injury_row(_QB1_ID, "QB", "Out", _LOCK + timedelta(hours=1)),
            ]
        )
        builder = _builder(injuries)

        selection = builder.fenced_injuries(_game_row(games, _GAME_ID), _LOCK)

        assert len(selection.rows) == 1, "one report per player: the latest admitted"
        assert selection.rows.iloc[0]["report_status"] == "Questionable"
        assert selection.information_time == _LOCK - timedelta(hours=1)

        frame = builder.build_features(
            games, _AS_OF_LATE, target_season=2023, target_week=1
        )
        assert float(frame.iloc[0]["home_qb_out_flag"]) == 0.0, (
            "the lock-2d Out was superseded by the lock-1h Questionable; the lock+1h Out "
            "was never known at the lock"
        )


# ---------------------------------------------------------------------------
# 3. Undatable rows are unknown with a flag, never known at the lock
# ---------------------------------------------------------------------------


class TestUndatableRows:
    def test_a_null_date_modified_row_is_excluded_and_its_game_flagged(self) -> None:
        games = _games()
        builder = _builder(_injuries([_injury_row(_QB1_ID, "QB", "Out", None)]))

        selection = builder.fenced_injuries(_game_row(games, _GAME_ID), _LOCK)
        frame = builder.build_features(
            games, _AS_OF_LATE, target_season=2023, target_week=1
        )
        provenance = builder.information_times(games, target_season=2023, target_week=1)

        assert selection.rule is SelectionRule.NONE_ADMITTED
        assert float(frame.iloc[0]["home_qb_out_flag"]) == 0.0
        assert float(frame.iloc[0]["home_injury_coverage"]) == 0.0
        assert float(frame.iloc[0]["home_date_modified_coverage"]) == 0.0
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value
        assert pd.isna(provenance.iloc[0]["information_time"])

    def test_an_undatable_row_beside_a_dated_one_still_flags_the_team(self) -> None:
        games = _games()
        builder = _builder(
            _injuries(
                [
                    _injury_row(
                        _QB1_ID, "QB", "Questionable", _LOCK - timedelta(days=1)
                    ),
                    _injury_row("00-0000009", "WR", "Out", None),
                ]
            )
        )

        frame = builder.build_features(
            games, _AS_OF_LATE, target_season=2023, target_week=1
        )

        assert float(frame.iloc[0]["home_injury_coverage"]) == 1.0
        assert float(frame.iloc[0]["home_date_modified_coverage"]) == 0.0, (
            "a dropped undatable row means the dated coverage is incomplete"
        )


# ---------------------------------------------------------------------------
# 4. The 2025-shaped frame and the capture-stamp column
# ---------------------------------------------------------------------------


def _frame_2025(stamp: pd.Timestamp | None) -> pd.DataFrame:
    """The 2025+ upstream shape: NO date_modified column; optionally the capture column."""
    frame = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "team": "KC",
                "gsis_id": _QB1_ID,
                "position": "QB",
                "full_name": "Patrick Mahomes",
                "report_status": "Out",
            }
        ]
    )
    if stamp is not None:
        frame[UPSTREAM_CAPTURE_COLUMN] = pd.Series([stamp]).dt.tz_convert("UTC")
    return frame


def _build_2025(stamp: pd.Timestamp | None):
    games = _games(season=2025)
    builder = InjuryBuilder(
        snap_builder=_snap_builder(),
        injuries_df=_frame_2025(stamp),
        depth_charts_df=_depth_charts(2025),
        pbp_df=pd.DataFrame(),
    )
    frame = builder.build_features(
        games, _AS_OF_LATE, target_season=2025, target_week=1
    )
    provenance = builder.information_times(games, target_season=2025, target_week=1)
    selection = builder.fenced_injuries(
        games.iloc[0].to_dict(), lock_rule.lock_frame(games).iloc[0]
    )
    return builder, games, frame, provenance, selection


class TestThe2025ShapedFrame:
    def test_no_date_modified_and_no_capture_column_is_an_honest_unknown(self) -> None:
        builder, games, frame, provenance, selection = _build_2025(None)

        assert selection.rule is SelectionRule.NONE_ADMITTED
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value
        assert pd.isna(provenance.iloc[0]["information_time"])
        assert float(frame.iloc[0]["home_injury_coverage"]) == 0.0
        assert float(frame.iloc[0]["home_qb_out_flag"]) == 0.0, (
            "the Out row was NOT known at the lock; no quiet week join may admit it"
        )
        for column, declared in builder.no_information_signature().items():
            assert float(frame.iloc[0][column]) == declared

    def test_the_gate_accepts_the_2025_shaped_frame(self) -> None:
        """The case that proves rung 5's full-history rebuild can complete."""
        builder, games, frame, provenance, _ = _build_2025(None)
        state = InformationTimeGate().check(
            "injury",
            frame,
            provenance,
            lock_rule.lock_frame(games),
            no_information_signature=builder.no_information_signature(),
        )
        assert state is SourceCheckState.CHECKED

    def test_a_capture_stamp_at_the_lock_is_admitted_on_the_capture_rule(self) -> None:
        lock = pd.Timestamp(lock_rule.game_lock(_KICKOFF.replace(year=2025)))
        builder, games, frame, provenance, selection = _build_2025(lock)

        assert selection.rule is SelectionRule.CAPTURE_STAMP
        assert provenance.iloc[0]["basis"] == InformationBasis.PER_ROW.value
        assert pd.Timestamp(provenance.iloc[0]["information_time"]) == lock
        assert float(frame.iloc[0]["home_qb_out_flag"]) == 1.0
        assert float(frame.iloc[0]["home_date_modified_coverage"]) == 0.0, (
            "a capture stamp is not a per-row date_modified"
        )

    def test_a_capture_stamp_one_second_after_the_lock_is_not_admitted(self) -> None:
        lock = pd.Timestamp(lock_rule.game_lock(_KICKOFF.replace(year=2025)))
        _, _, frame, provenance, selection = _build_2025(lock + _ONE_SECOND)

        assert selection.rule is SelectionRule.NONE_ADMITTED
        assert provenance.iloc[0]["basis"] == InformationBasis.NO_INFORMATION.value
        assert float(frame.iloc[0]["home_qb_out_flag"]) == 0.0

    def test_the_capture_column_name_is_the_documented_input_contract(self) -> None:
        assert UPSTREAM_CAPTURE_COLUMN == "upstream_captured_at"


# ---------------------------------------------------------------------------
# 5. The provenance is never dishonest -- UndatedSourceError is unreachable
# ---------------------------------------------------------------------------


def _every_case() -> list[tuple[InjuryBuilder, pd.DataFrame]]:
    lock_2025 = pd.Timestamp(lock_rule.game_lock(_KICKOFF.replace(year=2025)))
    cases: list[tuple[InjuryBuilder, pd.DataFrame]] = []
    for stamp in (None, lock_2025, lock_2025 + _ONE_SECOND):
        builder, games, *_ = _build_2025(stamp)
        cases.append((builder, games))
    for rows in (
        [_injury_row(_QB1_ID, "QB", "Out", _LOCK)],
        [_injury_row(_QB1_ID, "QB", "Out", _LOCK + _ONE_SECOND)],
        [_injury_row(_QB1_ID, "QB", "Out", None)],
    ):
        cases.append((_builder(_injuries(rows)), _games()))
    return cases


class TestTheProvenanceIsNeverDishonest:
    def test_no_per_row_row_ever_carries_a_null_time(self) -> None:
        for builder, games in _every_case():
            season = int(games["season"].iloc[0])
            provenance = builder.information_times(
                games, target_season=season, target_week=1
            )
            assert list(provenance.columns) == list(PROVENANCE_COLUMNS)
            per_row = provenance.loc[
                provenance["basis"] == InformationBasis.PER_ROW.value
            ]
            assert not per_row["information_time"].isna().any(), (
                "a per_row row with a null time is the dishonest declaration the gate "
                "refuses with UndatedSourceError; this builder must never emit one"
            )

    def test_the_gate_accepts_every_case_so_undated_refusal_is_unreachable(
        self,
    ) -> None:
        for builder, games in _every_case():
            season = int(games["season"].iloc[0])
            frame = builder.build_features(
                games, _AS_OF_LATE, target_season=season, target_week=1
            )
            provenance = builder.information_times(
                games, target_season=season, target_week=1
            )
            state = InformationTimeGate().check(
                "injury",
                frame,
                provenance,
                lock_rule.lock_frame(games),
                no_information_signature=builder.no_information_signature(),
            )
            assert state is SourceCheckState.CHECKED

    def test_the_no_information_signature_is_non_empty(self) -> None:
        signature = _builder(_injuries([_injury_row(_QB1_ID, "QB", "Out", _LOCK)]))
        assert signature.no_information_signature(), (
            "an empty signature beside a no_information row is a self-granted exemption"
        )

    def test_the_builder_is_an_information_time_provider(self) -> None:
        builder = InjuryBuilder(snap_builder=SnapCountBuilder())
        assert isinstance(builder, InformationTimeProvider)


# ---------------------------------------------------------------------------
# 6. Status vocabulary (kept from Plan 28-05)
# ---------------------------------------------------------------------------


class TestInjuryStatusVocab:
    @pytest.mark.parametrize(
        ("status", "expected_flag"),
        [("Out", 1.0), ("Doubtful", 1.0), ("Questionable", 0.0), (None, 0.0)],
    )
    def test_report_status_vocabulary(self, status, expected_flag) -> None:
        """The {Out, Doubtful} derivation reads ``report_status``; the row is pre-lock, so
        status (not the fence) is the only thing under test."""
        injuries = _injuries(
            [_injury_row(_QB1_ID, "QB", status, _LOCK - timedelta(days=1))]
        )
        out = _build(injuries)
        assert float(out.iloc[0]["home_qb_out_flag"]) == expected_flag


# ---------------------------------------------------------------------------
# 7. The removal of the week-keyed fallback, scanned with the docstrings EXCLUDED
# ---------------------------------------------------------------------------

_FALLBACK_NAME = "date_modified_applied"


def week_fallback_hits(source: str) -> list[str]:
    """Non-docstring string constants and identifiers equal to the retired flag name.

    DOCSTRINGS ARE EXCLUDED BY NODE POSITION (the first statement of a module, class or
    function body). Comments never reach the AST. So a contract docstring that explains
    the deletion by name is not a hit, while ``frame["date_modified_applied"] = False`` is.
    """
    tree = ast.parse(source)
    docstring_ids = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_ids
            and node.value == _FALLBACK_NAME
        ):
            hits.append(f"constant@{node.lineno}")
        elif isinstance(node, ast.Name) and node.id == _FALLBACK_NAME:
            hits.append(f"name@{node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr == _FALLBACK_NAME:
            hits.append(f"attribute@{node.lineno}")
        elif isinstance(node, ast.arg) and node.arg == _FALLBACK_NAME:
            hits.append(f"arg@{node.lineno}")
    return hits


class TestTheWeekKeyedFallbackIsGone:
    def test_control_a_planted_assignment_is_flagged(self) -> None:
        planted = 'def f(frame):\n    frame["date_modified_applied"] = False\n'
        assert week_fallback_hits(planted) == ["constant@2"]

    def test_control_docstrings_naming_the_deletion_are_not_flagged(self) -> None:
        explained = (
            '"""The module once reported date_modified_applied=False; it was deleted."""\n'
            "\n"
            "class Builder:\n"
            "    def fenced_injuries(self):\n"
            '        """Replaces the week-keyed fallback (date_modified_applied) with a\n'
            '        refusal to admit an undatable row."""\n'
            "        return None\n"
        )
        assert week_fallback_hits(explained) == []

    def test_the_injury_module_carries_no_code_reference_to_the_flag(self) -> None:
        assert week_fallback_hits(inspect.getsource(injury_module)) == []

    def test_the_selector_no_longer_returns_the_flag_in_a_tuple(self) -> None:
        tree = ast.parse(inspect.getsource(injury_module))
        selector = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fenced_injuries"
        ]
        assert len(selector) == 1
        returns = selector[0].returns
        assert returns is not None
        assert ast.unparse(returns) != "tuple[pd.DataFrame, bool]"

    def test_the_module_imports_nothing_from_the_future_stamp_module(self) -> None:
        tree = ast.parse(inspect.getsource(injury_module))
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any(m.startswith("data.upstream_asset_stamp") for m in imported)
