"""The per-game lock is a BINDING REFUSAL at selection time, not a descriptive value.

Phase 33, Plan 33-05 Task 1 (COLD-03, R6, D33-27/28); re-expressed on the day-before lock by
Plan 33.2-02 (D33.2-01, D33.2-18).

WHAT WAS MISSING AND WHY IT MATTERS
-----------------------------------
``backtest.weekly_bet_list._is_frozen`` only protects an ALREADY-STORED row from being
overwritten during an upsert. Nothing refused to EMIT a row decided after its game's lock, so a
row selected late would carry a post-hoc pick wearing a pre-game timestamp -- the repudiation
failure COLD-03 exists to prevent (T-33-21).

THE REFUSAL'S SUBJECT AND DIRECTION BOTH CHANGED, DELIBERATELY (Plan 33.2-02)
----------------------------------------------------------------------------
The fence used to refuse when the RUN started at or after the freeze (``now >= freeze``). Under
D33.2-18 the daily run captures before the lock and builds after it, so a run STARTING after a
lock is normal; what is refused now is a DECISION whose inputs were captured after the lock,
``decided_at > lock``. At-lock information is admissible under D33.2-01, so a decision exactly
AT the lock now yields a row. The three clock positions below -- one second after, exactly at,
one second before -- are the whole content of that claim, and the at-lock case is the one whose
verdict flipped.

WHY A STRICT PARSE WRAPPER RATHER THAN A WIDENED PARSER (D33-27, T-33-23)
------------------------------------------------------------------------
``scripts.ingest_historical_odds.normalize_snapshot_ts`` deliberately anchors a NAIVE value
in EASTERN and argues that at length in its own docstring: an unqualified wall-clock time in
this project's odds data is a market-local time. The freeze path needs the opposite -- a naive
value must RAISE, because a freeze comparison has no defensible default timezone. One function
cannot hold both behaviours without becoming a second answer wearing one name, so the strict
behaviour is a WRAPPER (``require_aware_snapshot_ts``) and the single-source property is
preserved by composition.

ALL THREE NAIVE SHAPES ARE DELIBERATELY CONSTRUCTED. Every one of the 234 stored bet-list
rows carries a tz-aware Eastern ``freeze_ts`` string, so the naive branch is UNREACHABLE from
real data -- which is exactly why it is safe to convert from a silent UTC assumption into a
raise now, and exactly why the inputs have to be built by hand.

Run this module:  uv run pytest tests/unit/test_freeze_fence_binding.py -q

Plain unit module: it reads nothing under ``data/``, ``artifacts/`` or ``outputs/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest.weekly_bet_list import (
    LockPassedError,
    select_games_for_decision_instant,
)
from scripts.ingest_historical_odds import (
    NaiveTimestampError,
    gameday_lock,
    normalize_snapshot_ts,
    require_aware_snapshot_ts,
)

EASTERN = ZoneInfo("America/New_York")

# Week 2's Sunday slate locks at Saturday 2026-09-19 18:00 ET = 22:00 UTC, MEASURED through the
# one rule rather than assumed. The Monday game (Sunday lock) and the week-3 Thursday game
# (Wednesday lock) each lock at a DIFFERENT instant, which is what makes them scoping controls.
_WEEK_2_SUNDAY = "2026-09-20"
_WEEK_2_MONDAY = "2026-09-21"
_WEEK_3_THURSDAY = "2026-09-24"
_WEEK_2_SUNDAY_LOCK_UTC = "2026-09-19T22:00:00+00:00"

# One second, the granularity the three clock positions are separated by. A whole second
# rather than a microsecond here because the claim is about the OPERATOR (>= against >), and a
# second is unambiguous in every representation the row can be stored in.
_ONE_SECOND = timedelta(seconds=1)


def _schedule() -> pd.DataFrame:
    """A constructed slate: two Sunday games sharing one lock, plus two other-lock controls.

    Week 2's two Sunday games both take the Saturday 2026-09-19 18:00 ET lock; the Monday game
    and the week-3 Thursday game lock on other days. The sharing is asserted from the one rule
    rather than hand-typed.
    """
    return pd.DataFrame(
        [
            {
                "game_id": "2026_02_CAR_ATL",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_SUNDAY,
            },
            {
                "game_id": "2026_02_SEA_KC",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_SUNDAY,
            },
            {
                "game_id": "2026_02_LV_NYJ",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_MONDAY,
            },
            {
                "game_id": "2026_03_ATL_GB",
                "season": 2026,
                "week": 3,
                "gameday": _WEEK_3_THURSDAY,
            },
        ]
    )


def _week_2_sunday_lock() -> datetime:
    return gameday_lock(_WEEK_2_SUNDAY)


_SUNDAY_GAMES = {"2026_02_CAR_ATL", "2026_02_SEA_KC"}


# ---------------------------------------------------------------------------
# The strict parse wrapper: three naive shapes, each deliberately constructed.
# ---------------------------------------------------------------------------


class TestANaiveTimestampRaisesRatherThanBeingAssumed:
    """T-33-23: a naive value is assumed neither UTC (today's bug) nor Eastern (the parser)."""

    def test_a_naive_datetime_raises_and_the_message_names_the_value_and_its_type(
        self,
    ) -> None:
        naive = datetime(2026, 9, 17, 18, 0)
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts(naive)

        message = str(excinfo.value)
        assert "2026-09-17 18:00:00" in message
        assert "datetime" in message
        assert "timezone" in message.lower()

    def test_a_naive_pandas_timestamp_raises(self) -> None:
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts(pd.Timestamp("2026-09-17 18:00:00"))

        message = str(excinfo.value)
        assert "2026-09-17 18:00:00" in message
        assert "Timestamp" in message

    def test_a_naive_iso_string_raises(self) -> None:
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts("2026-09-17T18:00:00")

        message = str(excinfo.value)
        assert "2026-09-17T18:00:00" in message
        assert "str" in message

    def test_the_message_says_a_freeze_comparison_has_no_defensible_default(
        self,
    ) -> None:
        """The reason travels with the refusal, so the next reader does not add a default."""
        with pytest.raises(NaiveTimestampError) as excinfo:
            require_aware_snapshot_ts("2026-09-17 18:00:00")
        assert "defensible" in str(excinfo.value)


class TestEveryAwareShapeTheParserDocumentsStillResolves:
    """The wrapper narrows NOTHING except naiveness: every aware shape round-trips."""

    @pytest.mark.parametrize(
        "value",
        [
            # The legacy per-season string.
            "2021-09-19T18:00:00-04:00",
            # The single non-consensus row's space-separated UTC string.
            "2025-09-29 18:44:09.707942+00:00",
            # A tz-aware datetime.
            datetime(2026, 9, 18, 18, 0, tzinfo=EASTERN),
            # A tz-aware pandas Timestamp.
            pd.Timestamp("2026-09-18T18:00:00-04:00"),
        ],
    )
    def test_an_aware_value_returns_a_utc_aware_datetime(self, value: object) -> None:
        resolved = require_aware_snapshot_ts(value)

        assert resolved.tzinfo is not None
        assert resolved.utcoffset() == timedelta(0)
        # The same INSTANT the one parse path resolves, only expressed in UTC.
        assert resolved == normalize_snapshot_ts(value)

    def test_the_wrapper_delegates_rather_than_reimplementing_the_parse(self) -> None:
        """Every aware shape agrees with ``normalize_snapshot_ts`` to the microsecond."""
        value = "2025-09-29 18:44:09.707942+00:00"
        assert require_aware_snapshot_ts(value) == normalize_snapshot_ts(value)
        assert require_aware_snapshot_ts(value).isoformat().endswith("+00:00")

    @pytest.mark.parametrize("value", [None, "", float("nan"), pd.NaT])
    def test_a_null_or_empty_value_keeps_the_parsers_own_refusal(
        self, value: object
    ) -> None:
        """CONTROL: the wrapper adds a naiveness rule and takes nothing away.

        A null is still the parser's ``ValueError`` and is NOT relabelled as naive -- a
        missing instant and an unqualified one are different failures with different fixes.
        """
        with pytest.raises(ValueError) as excinfo:
            require_aware_snapshot_ts(value)
        assert not isinstance(excinfo.value, NaiveTimestampError)


# ---------------------------------------------------------------------------
# The binding selection fence: three clock positions.
# ---------------------------------------------------------------------------


class TestSelectionRefusesADecisionAfterTheLock:
    """R6: a post-lock decision is refused BY NAME -- never emitted, never silently dropped."""

    def test_the_measured_lock_is_saturday_evening_eastern(self) -> None:
        assert _week_2_sunday_lock().isoformat() == "2026-09-19T18:00:00-04:00"
        assert require_aware_snapshot_ts(_week_2_sunday_lock()).isoformat() == (
            _WEEK_2_SUNDAY_LOCK_UTC
        )

    def test_one_second_after_the_lock_selection_raises_naming_the_game(self) -> None:
        instant = _week_2_sunday_lock()
        with pytest.raises(LockPassedError) as excinfo:
            select_games_for_decision_instant(
                _schedule(), instant, decided_at=instant + _ONE_SECOND
            )

        message = str(excinfo.value)
        assert "2026_02_CAR_ATL" in message
        # BOTH instants are named, so the reader can see which side of the fence it is on.
        assert require_aware_snapshot_ts(instant).isoformat() in message
        assert require_aware_snapshot_ts(instant + _ONE_SECOND).isoformat() in message

    def test_exactly_at_the_lock_the_row_is_emitted(self) -> None:
        """The equality edge, stated explicitly, and it is the verdict that FLIPPED.

        At-lock information is admissible (D33.2-01), so a decision whose inputs were captured
        exactly at the lock is a legitimate decision. The retired fence refused this instant;
        a test that only covered one-second-after would pass under either operator and would
        therefore assert nothing about the edge.
        """
        instant = _week_2_sunday_lock()
        selected = select_games_for_decision_instant(
            _schedule(), instant, decided_at=instant
        )
        assert set(selected["game_id"]) == _SUNDAY_GAMES

    def test_one_second_before_the_lock_the_row_is_emitted(self) -> None:
        instant = _week_2_sunday_lock()
        selected = select_games_for_decision_instant(
            _schedule(), instant, decided_at=instant - _ONE_SECOND
        )
        assert set(selected["game_id"]) == _SUNDAY_GAMES

    def test_the_run_start_is_not_an_input_only_the_decision_instant_is(self) -> None:
        """A run that STARTS after the lock but decided on pre-lock inputs still emits.

        The daily run captures before the lock and builds after it (D33.2-18), so its start
        is routinely past the lock. The fence takes no run clock at all -- only the explicit
        decision instant -- so a late start cannot turn a clean decision into a refusal.
        """
        import inspect

        params = set(inspect.signature(select_games_for_decision_instant).parameters)
        assert "decided_at" in params
        assert "now" not in params, "the fence must not read the run's own clock"

        instant = _week_2_sunday_lock()
        captured_before_lock = instant - timedelta(minutes=30)
        selected = select_games_for_decision_instant(
            _schedule(), instant, decided_at=captured_before_lock
        )
        assert set(selected["game_id"]) == _SUNDAY_GAMES

    def test_the_refusal_is_not_catchable_as_a_value_error(self) -> None:
        """LockPassedError stays a RuntimeError, outside the absent-input ValueError tuple."""
        instant = _week_2_sunday_lock()
        assert issubclass(LockPassedError, RuntimeError)
        assert not issubclass(LockPassedError, ValueError)

        caught_as_value_error = False
        with pytest.raises(LockPassedError):
            try:
                select_games_for_decision_instant(
                    _schedule(), instant, decided_at=instant + _ONE_SECOND
                )
            except ValueError:
                caught_as_value_error = True
        assert caught_as_value_error is False

    def test_a_game_belonging_to_a_different_instant_is_simply_not_selected(
        self,
    ) -> None:
        """Scoping is not refusing. Monday and Thursday games are OUT OF SCOPE here.

        They are absent because their locks are DIFFERENT instants, not because the fence
        refused them -- so nothing raises. Keeping the two outcomes distinct is what makes
        ``LockPassedError`` a tripwire rather than an ordinary control-flow signal.
        """
        instant = _week_2_sunday_lock()
        selected = select_games_for_decision_instant(
            _schedule(), instant, decided_at=instant
        )
        assert "2026_02_LV_NYJ" not in set(selected["game_id"])
        assert "2026_03_ATL_GB" not in set(selected["game_id"])

    def test_a_naive_decision_instant_raises_rather_than_being_assumed_utc(
        self,
    ) -> None:
        """Both sides of the comparison go through the strict wrapper (R6 timezone edge)."""
        naive_decision = datetime(2026, 9, 19, 17, 59)
        with pytest.raises(NaiveTimestampError):
            select_games_for_decision_instant(
                _schedule(), _week_2_sunday_lock(), decided_at=naive_decision
            )

    def test_the_per_game_lock_is_never_re_derived_here(self) -> None:
        """The selected set is exactly the set the ONE lock rule assigns to this instant.

        Asserted against ``gameday_lock`` (which hands the date to ``utils.game_lock``) per row,
        rather than against hand-written date arithmetic, so a second rule inside the selection
        would show up as a disagreement.
        """
        schedule = _schedule()
        instant = _week_2_sunday_lock()
        expected = {
            str(row.game_id)
            for row in schedule.itertuples()
            if gameday_lock(str(row.gameday)) == instant
        }
        selected = select_games_for_decision_instant(
            schedule, instant, decided_at=instant
        )
        assert set(selected["game_id"]) == expected
        assert expected == _SUNDAY_GAMES, (
            "the fixture's lock sharing is not what it claims"
        )


class TestExcludedGamesAreDroppedBeforeTheCheck:
    """D33.2-05: a game the run chose to skip is removed from scope BEFORE the refusal."""

    def test_an_excluded_game_is_absent_and_the_clean_game_is_kept(self) -> None:
        instant = _week_2_sunday_lock()
        selected = select_games_for_decision_instant(
            _schedule(),
            instant,
            decided_at=instant,
            excluded_game_ids=frozenset({"2026_02_CAR_ATL"}),
        )
        assert set(selected["game_id"]) == {"2026_02_SEA_KC"}

    def test_an_excluded_game_past_its_lock_does_not_raise(self) -> None:
        """A game skipped for a post-lock input is usually a game whose lock has passed.

        Checking before excluding would raise for a game the run had already decided not to
        bet, turning one clean skip into a failure that costs the day's clean games too.
        """
        instant = _week_2_sunday_lock()
        selected = select_games_for_decision_instant(
            _schedule(),
            instant,
            decided_at=instant + _ONE_SECOND,
            excluded_game_ids=frozenset(_SUNDAY_GAMES),
        )
        assert selected.empty

    def test_the_default_is_an_empty_frozenset_and_changes_nothing(self) -> None:
        import inspect

        param = inspect.signature(select_games_for_decision_instant).parameters[
            "excluded_game_ids"
        ]
        assert param.default == frozenset()
        assert param.kind is inspect.Parameter.KEYWORD_ONLY

        instant = _week_2_sunday_lock()
        with_default = select_games_for_decision_instant(
            _schedule(), instant, decided_at=instant
        )
        explicit_empty = select_games_for_decision_instant(
            _schedule(), instant, decided_at=instant, excluded_game_ids=frozenset()
        )
        pd.testing.assert_frame_equal(with_default, explicit_empty)


# ---------------------------------------------------------------------------
# The back-compat READ shim.
# ---------------------------------------------------------------------------
#
# WHY THESE LIVE HERE. Plan 33-05 Task 1 creates ``read_bet_list_with_schema_shim`` but names no
# test module for it -- its three named modules are all about the fence. It is filed under this
# module, the task's primary home, rather than under ``tests/unit/test_bet_list_schema.py``,
# which Task 3 owns and which cannot be extended before the owner checkpoint that locks the
# column count has been ruled on.


class TestTheBackCompatReadShim:
    """A 28-column parquet still reads after the schema goes to 29 -- and is never rewritten."""

    def _pre_bump_columns(self) -> list[str]:
        """The 28 columns a stored file carries, derived rather than copied.

        Taken as ``BET_LIST_READ_COLUMNS`` minus the one column this phase adds, so this list is
        the pre-bump order BOTH before and after ``api.cache`` carries it. Deriving it from
        ``BET_LIST_COLUMNS`` instead would silently become the 29-column order after the bump,
        and the shim would then be handed a frame that needs no shimming -- a test that passes by
        no longer exercising the thing it names.
        """
        from backtest.weekly_bet_list import BET_LIST_READ_COLUMNS, DECIDED_AT_COLUMN

        return [c for c in BET_LIST_READ_COLUMNS if c != DECIDED_AT_COLUMN]

    def _stored_28_column_parquet(self, tmp_path: Path) -> Path:
        from backtest.weekly_bet_list import BET_LIST_ARTIFACT_NAME

        columns = self._pre_bump_columns()
        assert len(columns) == 28, len(columns)
        frame = pd.DataFrame(
            [dict.fromkeys(columns)],
            columns=pd.Index(columns),
        )
        frame["game_id"] = "2025_W01_DET@KC"
        frame["provenance"] = "backtest_replay"
        path = tmp_path / BET_LIST_ARTIFACT_NAME
        frame.to_parquet(path, index=False)
        return path

    def test_a_28_column_file_reads_back_as_29_with_a_null_observation_time(
        self, tmp_path: Path
    ) -> None:
        from backtest.weekly_bet_list import (
            BET_LIST_READ_COLUMNS,
            DECIDED_AT_COLUMN,
            read_bet_list_with_schema_shim,
        )

        path = self._stored_28_column_parquet(tmp_path)
        shimmed = read_bet_list_with_schema_shim(path)

        assert list(shimmed.columns) == list(BET_LIST_READ_COLUMNS)
        assert len(shimmed.columns) == 29
        assert shimmed[DECIDED_AT_COLUMN].isna().all(), (
            "the shim invented an observation time for a row that was never observed"
        )

    def test_the_shim_writes_nothing(self, tmp_path: Path) -> None:
        """The plan's own named prohibition, asserted on the file's BYTES.

        Every stored row is already past its freeze, so stamping an observation time onto one
        now would record a time at which nobody observed anything.
        """
        import hashlib

        from backtest.weekly_bet_list import read_bet_list_with_schema_shim

        path = self._stored_28_column_parquet(tmp_path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        read_bet_list_with_schema_shim(path)
        read_bet_list_with_schema_shim(path)
        after = hashlib.sha256(path.read_bytes()).hexdigest()

        assert after == before, "the READ shim rewrote the stored ledger"
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]

    def test_a_column_other_than_the_new_one_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """Only ``decided_at_utc`` is filled. A file written by a different schema is refused."""
        from backtest.weekly_bet_list import (
            BET_LIST_ARTIFACT_NAME,
            read_bet_list_with_schema_shim,
        )

        columns = [c for c in self._pre_bump_columns() if c != "ev_tier"]
        path = tmp_path / BET_LIST_ARTIFACT_NAME
        pd.DataFrame([dict.fromkeys(columns)], columns=pd.Index(columns)).to_parquet(
            path, index=False
        )

        with pytest.raises(ValueError, match="missing column"):
            read_bet_list_with_schema_shim(path)

    def test_the_one_stored_artifact_reader_is_routed_through_the_shim(self) -> None:
        """``data.graded_weeks`` and the cache-source reader both come through this one seam."""
        import inspect

        from backtest import weekly_bet_list

        source = inspect.getsource(weekly_bet_list.read_bet_list_artifact)
        assert "read_bet_list_with_schema_shim" in source
        assert "pd.read_parquet" not in source, (
            "read_bet_list_artifact reads the parquet itself again, so the shim is bypassed"
        )
